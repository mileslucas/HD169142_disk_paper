import proplot as pro
import numpy as np
import paths
from astropy.io import fits
from skimage.transform import warp_polar
from astropy.convolution import convolve, kernels
import tqdm
from astropy.visualization import simple_norm
from target_info import target_info
from scipy import interpolate
import pandas as pd

from utils_organization import folders, pxscales, time_from_folder
from utils_ephemerides import keplerian_warp
from utils_plots import setup_rc
from utils_errorprop import bootstrap_argmax_and_max, bootstrap_argmin_and_min
from matplotlib.transforms import blended_transform_factory

if __name__ == "__main__": 
    setup_rc()

    alma_folder = "20170918_ALMA_1.3mm"
    alma_table = pd.read_csv(paths.data / alma_folder / f"{alma_folder}_HD169142_radial_profiles.csv")
    alma_radii = alma_table["radius(au)"].values
    alma_curve = alma_table["I"].values
    alma_err = alma_table["I_err"].values
    alma_time = time_from_folder(alma_folder)
    
    ## Plot and save
    width = 3.31314
    aspect_ratio = 1/1.5
    height = width * aspect_ratio
    fig, axes = pro.subplots(
        nrows=1, width=f"{width}in", height=f"{height}in", hspace=0.25
    )

    common_rs = np.linspace(0, alma_radii.max(), 2 * len(alma_radii))
    curves = []
    errs = []
    _folders = folders.copy()
    _idx = _folders.index("20230604_CHARIS_JHK")
    del _folders[_idx]
    
    for i, folder in enumerate(tqdm.tqdm(folders)):

        this_time = time_from_folder(folder)
        table = pd.read_csv(paths.data / folder / f"{folder}_HD169142_radial_profiles.csv")
        itp_values = interpolate.CubicSpline(table["radius(au)"].values, table["Qphi"].values)(common_rs)
        itp_errs = interpolate.CubicSpline(table["radius(au)"].values, table["Qphi_err"].values)(common_rs)
        norm_val = itp_values.max()
        curves.append(itp_values / norm_val)
        errs.append(itp_errs / norm_val)

    mean_curve = np.nanmean(curves, axis=0)
    stderr_curve = np.nanstd(curves, axis=0) / len(curves)
    rmserr_curve = np.sqrt(np.nansum(np.power(errs, 2), axis=0) / len(errs)**2)
    err_curve = np.hypot(stderr_curve, rmserr_curve)
    
    norm_val = np.nanmax(mean_curve)
    mean_curve /= norm_val
    err_curve /= norm_val


    with open(paths.data / "radial_profile_peaks.csv", "w") as fh:
        qphi_peak, qphi_peak_err, _, _ = bootstrap_argmax_and_max(common_rs, mean_curve, err_curve)
        print(f"Qphi inner peak: {qphi_peak} ± {qphi_peak_err} au")
        fh.write(f"Qphi,inner,{qphi_peak},{qphi_peak_err}0,0\n")

        outer_mask = (45 <= common_rs) & (common_rs <= 100)
        outer_peak, outer_peak_err, _, _ = bootstrap_argmax_and_max(common_rs[outer_mask], mean_curve[outer_mask], err_curve[outer_mask])
        print(f"Qphi outer peak: {outer_peak} ± {outer_peak_err} au")
        fh.write(f"Qphi,outer,{outer_peak},{outer_peak_err}0,0\n")
        
        alma_peak, alma_peak_err, alma_peak_mJy, alma_peak_err_mJy = bootstrap_argmax_and_max(alma_radii, alma_curve, alma_err)
        print(f"ALMA B1 peak: {alma_peak} ± {alma_peak_err} au, {alma_peak_mJy} ± {alma_peak_err_mJy} mJy")
        fh.write(f"ALMA,B1,{alma_peak},{alma_peak_err},{alma_peak_mJy},{alma_peak_err_mJy}\n")

        b2_mask = (50 <= alma_radii) & (alma_radii <= 61)
        b2_peak, b2_peak_err, b2_peak_mJy, b2_peak_err_mJy = bootstrap_argmax_and_max(alma_radii[b2_mask], alma_curve[b2_mask], alma_err[b2_mask])
        print(f"ALMA B2 peak: {b2_peak} ± {b2_peak_err} au, {b2_peak_mJy} ± {b2_peak_err_mJy} mJy")
        fh.write(f"ALMA,B2,{b2_peak},{b2_peak_err},{b2_peak_mJy},{b2_peak_err_mJy}\n")

        b3_mask = (61 <= alma_radii) & (alma_radii <= 70)
        b3_peak, b3_peak_err, b3_peak_mJy, b3_peak_err_mJy = bootstrap_argmax_and_max(alma_radii[b3_mask], alma_curve[b3_mask], alma_err[b3_mask])
        print(f"ALMA B3 peak: {b3_peak} ± {b3_peak_err} au, {b3_peak_mJy} ± {b3_peak_err_mJy} mJy")
        fh.write(f"ALMA,B3,{b3_peak},{b3_peak_err},{b3_peak_mJy},{b3_peak_err_mJy}\n")

        b4_mask = (70 <= alma_radii) & (alma_radii <= 85)
        b4_peak, b4_peak_err, b4_peak_mJy, b4_peak_err_mJy = bootstrap_argmax_and_max(alma_radii[b4_mask], alma_curve[b4_mask], alma_err[b4_mask])
        print(f"ALMA B4 peak: {b4_peak} ± {b4_peak_err} au, {b4_peak_mJy} ± {b4_peak_err_mJy} mJy")
        fh.write(f"ALMA,B4,{b4_peak},{b4_peak_err},{b4_peak_mJy},{b4_peak_err_mJy}\n")






    axes[0].plot(common_rs, mean_curve, shadedata=err_curve, c="C0", label=r"Mean $Q_\phi \times r^2$", zorder=10)
    axes[0].plot(alma_radii, alma_curve / alma_peak_mJy, shadedata=alma_err / alma_peak_mJy, c="C3", label="ALMA (1.3mm)", zorder=9)


    axes[0].fill_betweenx(axes[0].get_ylim(), qphi_peak - qphi_peak_err, qphi_peak + qphi_peak_err, c="C0", zorder=0, lw=0, alpha=0.2)
    axes[0].fill_betweenx(axes[0].get_ylim(), outer_peak - outer_peak_err, outer_peak + outer_peak_err, c="C0", zorder=0, lw=0, alpha=0.2)
    axes[0].axvline(alma_peak, c="C3", zorder=0, lw=1, alpha=0.7)
    axes[0].axvline(b2_peak, c="C3", zorder=0, lw=1, alpha=0.7)
    axes[0].axvline(b3_peak, c="C3", zorder=0, lw=1, alpha=0.7)
    axes[0].axvline(b4_peak, c="C3", zorder=0, lw=1, alpha=0.7)


    axes[0].text(
        qphi_peak, 1.1, "Inner",
        c="C0",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )
    axes[0].text(
        outer_peak, 1.1, "Outer",
        c="C0",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )
    axes[0].text(
        0, 1.02, "B0",
        c="0.3",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )
    axes[0].text(
        alma_peak, 1.02, "B1",
        c="0.3",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )
    axes[0].text(
        b2_peak, 1.02, "B2",
        c="0.3",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )
    axes[0].text(
        b3_peak, 1.02, "B3",
        c="0.3",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )
    axes[0].text(
        b4_peak, 1.02, "B4",
        c="0.3",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )


    axes[0].legend(ncols=1)

    ## sup title
    axes.format(
        ylim=(-0.1, None),
        xlim=(0, 115),
        xlabel="Separation (au)",
        ylabel="Normalized profile",
    ) 


    mask = (common_rs >= 21) & (common_rs <= 65)

    r_min, r_min_err, I_rmin, I_rmin_err = bootstrap_argmin_and_min(common_rs[mask], mean_curve[mask], err_curve[mask])

    axes[0].axvline(r_min, c="0.3", ls="--", lw=1, alpha=0.7)
    print(f"Qphi gap r_min: {r_min} ± {r_min_err} au, {I_rmin * 100} ± {I_rmin_err * 100} %")

    r1 = 2/3 * r_min
    r2 = 3/2 * r_min
    I_r1 = mean_curve[np.where(common_rs >= r1)[0][0]]
    I_r2 = mean_curve[np.where(common_rs >= r2)[0][0]]
    I0_rmin = np.sqrt(I_r1 * I_r2)
    idx_min = np.where(common_rs >= r_min)[0][0]
    delta_I = I0_rmin / I_rmin # Eqn 12

    I_edge = np.sqrt(I0_rmin * I_rmin) # Eqn 13
    idx_in = np.where((mean_curve <= I_edge) & (common_rs > qphi_peak))[0][0]
    r_in = common_rs[idx_in]
    idx_out = np.where((mean_curve >= I_edge) & (common_rs > r_min))[0][0]
    r_out = common_rs[idx_out]
    # gap width in au
    w_I = r_out - r_in # Eqn 14
    norm_w_I = w_I / r_min # Eqn 15
    
    beta = 0.04 * target_info.dist_pc / w_I # Eqn 19
    # normalized surface density gap width
    norm_w_S = np.sqrt(norm_w_I**2 - 0.13 * beta**2 / delta_I) # Eqn 20
    aspect_ratio = norm_w_S / 5.8 # Eqn 16
    _coeff = np.power(delta_I, 1 / (0.85 - 0.44 * beta**2))
    delta_S = _coeff / (1 - 0.0069 * _coeff) 
    # delta_S = delta_I**(1 / (0.85 - 0.44*beta**2))
    alpha_test= np.array((1e-4, 1e-3, 1e-2))
    q = np.sqrt((delta_S - 1) * (aspect_ratio)**5 / 0.043 * alpha_test)
    Mp = q * target_info.stellar_mass * 1047
    
    print(f"Assuming Qphi disk β: {beta}")
    print(f"Qphi disk gap width: {w_I} au, ΔI: {norm_w_I}")
    print(f"Qphi disk h/r: {aspect_ratio}")
    print(f"Qphi disk density width: {norm_w_S * r_min} au, ΔI: {norm_w_S}")
    print(f"Qphi disk gap depth: {delta_I}, density depth: {delta_S}")

    print(f"Using ⍺: {alpha_test}")
    print(f"Qphi mass ratios: {q}")
    print(f"Qphi planet masses: {Mp} M_J")

    axes[0].fill_betweenx(axes[0].get_ylim(), r_min - norm_w_S * r_min/2, r_min + norm_w_S * r_min/2, c="0.3", lw=0, alpha=0.1)

    # axes[0].axvline(10, c="0.3", ls="--", lw=1)
    # axes[0].text(
    #     10, 1.02, "D1",
    #     c="0.3",
    #     transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
    #     fontsize=pro.rc["title.size"],
    #     weight="bold",
    #     ha="center",
    #     va="bottom",
    # )
    axes[0].text(
        r_min, 1.02, "D2",
        c="0.3",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )


    # axes[0].axhline(I0_rmin, c="0.3", lw=1, ls=":")
    # axes[0].axhline(I_rmin, c="0.3", lw=1, ls=":")
    # axes[0].axhline(I_edge, c="0.3", lw=1, ls="-.")

    # axes[0].axvline(r_in, c="0.3", lw=1, ls=":")
    # axes[0].axvline(r_out, c="0.3", lw=1, ls=":")

    fig.savefig(
        paths.figures / "HD169142_radial_profiles_Qphi_combined_ALMA.pdf", bbox_inches="tight"
    )


