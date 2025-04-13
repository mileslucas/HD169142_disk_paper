import numpy as np
import pandas as pd
import paths
import proplot as pro
import tqdm
from matplotlib.transforms import blended_transform_factory
from scipy import interpolate
from target_info import target_info
from utils_errorprop import bootstrap_argmax_and_max, bootstrap_interpolate
from utils_organization import folders, time_from_folder
from utils_plots import setup_rc


def calculate_gap_params(radii_au, profile):
    idx_min = np.argmin(profile)
    r_min = radii_au[idx_min]
    I_rmin = profile[idx_min]
    r1 = 2 / 3 * r_min
    r2 = 3 / 2 * r_min
    if r1 < radii_au.min():
        r1 = radii_au.min()
    if r2 > radii_au.max():
        r2 = radii_au.max()
    I_r1 = profile[np.where(radii_au >= r1)[0][0]]
    I_r2 = profile[np.where(radii_au >= r2)[0][0]]
    I0_rmin = np.sqrt(I_r1 * I_r2)
    idx_min = np.where(radii_au >= r_min)[0][0]
    delta_I = I0_rmin / I_rmin  # Eqn 12

    I_edge = np.sqrt(I0_rmin * I_rmin)  # Eqn 13
    idx_in = np.where((profile <= I_edge) & (radii_au > qphi_peak))[0][0]
    r_in = radii_au[idx_in]
    idx_out = np.where((profile >= I_edge) & (radii_au > r_min))[0][0]
    r_out = radii_au[idx_out]
    # gap width in au
    w_I = r_out - r_in  # Eqn 14
    norm_w_I = w_I / r_min  # Eqn 15

    beta = 0.04 * target_info.dist_pc / w_I  # Eqn 19
    # normalized surface density gap width
    norm_w_S = np.sqrt(norm_w_I**2 - 0.13 * beta**2 / delta_I)  # Eqn 20
    aspect_ratio = norm_w_S / 5.8  # Eqn 16
    _coeff = np.power(delta_I, 1 / (0.85 - 0.44 * beta**2))
    delta_S = _coeff / (1 - 0.0069 * _coeff)

    alpha_test = np.array((1e-4, 1e-3, 1e-2))
    q = np.sqrt((delta_S - 1) * (aspect_ratio) ** 5 / 0.043 * alpha_test)
    Mp = q * target_info.stellar_mass * 1047

    param_vec = [
        r_min,
        I_rmin,
        delta_I,
        w_I,
        norm_w_I,
        beta,
        norm_w_S,
        aspect_ratio,
        delta_S,
        *q,
        *Mp,
    ]
    return param_vec


def bootstrap_gap_params(radii_au, profile, profile_err, N=10000):
    signal_samples = old_y[None, :] + np.random.randn(N, len(old_y)) * old_err[None, :]
    rng = np.random.default_rng(169142)
    samples = rng.normal(loc=profile, scale=profile_err, size=(N, len(profile)))

    results = [calculate_gap_params(radii_au, sample) for sample in samples]

    mean_result = np.nanmean(results, axis=0)
    stderr_result = np.nanstd(results, axis=0)

    return mean_result, stderr_result


if __name__ == "__main__":
    setup_rc()

    alma_folder = "20170918_ALMA_1.3mm"
    alma_table = pd.read_csv(
        paths.data / alma_folder / f"{alma_folder}_HD169142_radial_profiles.csv"
    )
    alma_radii = alma_table["radius(au)"].values
    alma_curve = alma_table["I"].values
    alma_err = alma_table["I_err"].values
    alma_time = time_from_folder(alma_folder)

    ## Plot and save
    width = 3.31314
    aspect_ratio = 1 / 1.5
    height = width * aspect_ratio
    fig, axes = pro.subplots(nrows=1, width=f"{width}in", height=f"{height}in", hspace=0.25)

    common_rs = np.linspace(7, alma_radii.max(), len(alma_radii[alma_radii > 7]))
    curves = []
    errs = []
    _folders = folders.copy()
    _idx = _folders.index("20230604_CHARIS_JHK")
    del _folders[_idx]

    for _i, folder in enumerate(tqdm.tqdm(folders)):
        this_time = time_from_folder(folder)
        table = pd.read_csv(paths.data / folder / f"{folder}_HD169142_radial_profiles.csv")
        itp_values, itp_errs = bootstrap_interpolate(
            common_rs, table["radius(au)"].values, table["Qphi"].values, table["Qphi_err"].values
        )

        itp_values = interpolate.CubicSpline(table["radius(au)"].values, table["Qphi"].values)(
            common_rs
        )
        itp_errs = interpolate.CubicSpline(table["radius(au)"].values, table["Qphi_err"].values)(
            common_rs
        )
        norm_val = itp_values.max()
        curves.append(itp_values / norm_val)
        errs.append(itp_errs / norm_val)

    mean_curve = np.nanmean(curves, axis=0)
    stderr = np.nanstd(curves, axis=0) / np.sqrt(len(curves))
    rmserr = np.sqrt(np.sum(np.power(errs, 2), axis=0)) / len(curves)
    stderr_curve = np.hypot(stderr, rmserr)

    norm_val = np.nanmax(mean_curve)
    mean_curve /= norm_val
    stderr_curve /= norm_val

    with (paths.data / "radial_profile_peaks.csv").open("w") as fh:
        inner_mask = (common_rs >= 10) & (common_rs <= 40)
        qphi_peak, qphi_peak_err, _, _ = bootstrap_argmax_and_max(
            common_rs[inner_mask], mean_curve[inner_mask], stderr_curve[inner_mask]
        )
        print(f"Qphi inner peak: {qphi_peak} ± {qphi_peak_err} au")
        fh.write(f"Qphi,inner,{qphi_peak},{qphi_peak_err}0,0\n")

        outer_mask = (common_rs >= 45) & (common_rs <= 100)
        outer_peak, outer_peak_err, _, _ = bootstrap_argmax_and_max(
            common_rs[outer_mask], mean_curve[outer_mask], stderr_curve[outer_mask]
        )
        print(f"Qphi outer peak: {outer_peak} ± {outer_peak_err} au")
        fh.write(f"Qphi,outer,{outer_peak},{outer_peak_err}0,0\n")

        alma_peak, alma_peak_err, alma_peak_mJy, alma_peak_err_mJy = bootstrap_argmax_and_max(
            alma_radii, alma_curve, alma_err
        )
        print(
            f"ALMA B1 peak: {alma_peak} ± {alma_peak_err} au, {alma_peak_mJy} ± {alma_peak_err_mJy} mJy"
        )
        fh.write(f"ALMA,B1,{alma_peak},{alma_peak_err},{alma_peak_mJy},{alma_peak_err_mJy}\n")

        b2_mask = (alma_radii >= 50) & (alma_radii <= 61)
        b2_peak, b2_peak_err, b2_peak_mJy, b2_peak_err_mJy = bootstrap_argmax_and_max(
            alma_radii[b2_mask], alma_curve[b2_mask], alma_err[b2_mask]
        )
        print(f"ALMA B2 peak: {b2_peak} ± {b2_peak_err} au, {b2_peak_mJy} ± {b2_peak_err_mJy} mJy")
        fh.write(f"ALMA,B2,{b2_peak},{b2_peak_err},{b2_peak_mJy},{b2_peak_err_mJy}\n")

        b3_mask = (alma_radii >= 61) & (alma_radii <= 70)
        b3_peak, b3_peak_err, b3_peak_mJy, b3_peak_err_mJy = bootstrap_argmax_and_max(
            alma_radii[b3_mask], alma_curve[b3_mask], alma_err[b3_mask]
        )
        print(f"ALMA B3 peak: {b3_peak} ± {b3_peak_err} au, {b3_peak_mJy} ± {b3_peak_err_mJy} mJy")
        fh.write(f"ALMA,B3,{b3_peak},{b3_peak_err},{b3_peak_mJy},{b3_peak_err_mJy}\n")

        b4_mask = (alma_radii >= 70) & (alma_radii <= 85)
        b4_peak, b4_peak_err, b4_peak_mJy, b4_peak_err_mJy = bootstrap_argmax_and_max(
            alma_radii[b4_mask], alma_curve[b4_mask], alma_err[b4_mask]
        )
        print(f"ALMA B4 peak: {b4_peak} ± {b4_peak_err} au, {b4_peak_mJy} ± {b4_peak_err_mJy} mJy")
        fh.write(f"ALMA,B4,{b4_peak},{b4_peak_err},{b4_peak_mJy},{b4_peak_err_mJy}\n")

    axes[0].plot(
        common_rs,
        mean_curve,
        shadedata=stderr_curve,
        c="C0",
        label=r"Mean $Q_\phi \times r^2$",
        zorder=10,
    )
    axes[0].plot(
        alma_radii,
        alma_curve / alma_peak_mJy,
        shadedata=alma_err / alma_peak_mJy,
        c="C3",
        label="ALMA (1.3mm)",
        zorder=9,
    )

    axes[0].fill_betweenx(
        axes[0].get_ylim(),
        qphi_peak - qphi_peak_err,
        qphi_peak + qphi_peak_err,
        c="C0",
        zorder=0,
        lw=0,
        alpha=0.2,
    )
    axes[0].axvline(qphi_peak, c="C0", zorder=1, lw=1, alpha=0.7)
    axes[0].fill_betweenx(
        axes[0].get_ylim(),
        outer_peak - outer_peak_err,
        outer_peak + outer_peak_err,
        c="C0",
        zorder=0,
        lw=0,
        alpha=0.2,
    )
    axes[0].axvline(outer_peak, c="C0", zorder=1, lw=1, alpha=0.7)

    axes[0].axvline(alma_peak, c="C3", zorder=0, lw=1, alpha=0.7)
    axes[0].axvline(b2_peak, c="C3", zorder=0, lw=1, alpha=0.7)
    axes[0].axvline(b3_peak, c="C3", zorder=0, lw=1, alpha=0.7)
    axes[0].axvline(b4_peak, c="C3", zorder=0, lw=1, alpha=0.7)

    axes[0].text(
        qphi_peak,
        1.1,
        "Inner",
        c="C0",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )
    axes[0].text(
        outer_peak,
        1.1,
        "Outer",
        c="C0",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )
    axes[0].text(
        0,
        1.02,
        "B0",
        c="0.3",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )
    axes[0].text(
        alma_peak,
        1.02,
        "B1",
        c="0.3",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )
    axes[0].text(
        b2_peak,
        1.02,
        "B2",
        c="0.3",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )
    axes[0].text(
        b3_peak,
        1.02,
        "B3",
        c="0.3",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )
    axes[0].text(
        b4_peak,
        1.02,
        "B4",
        c="0.3",
        transform=blended_transform_factory(axes[0].transData, axes[0].transAxes),
        weight="bold",
        ha="center",
        va="bottom",
    )

    axes[0].legend(ncols=1)

    ## sup title
    axes.format(
        ylim=(-0.1, None), xlim=(0, 115), xlabel="Separation (au)", ylabel="Normalized profile"
    )

    mask = (common_rs >= 20) & (common_rs <= 65)

    params, param_errs = bootstrap_gap_params(common_rs[mask], mean_curve[mask], stderr_curve[mask])
    (
        r_min,
        I_rmin,
        delta_I,
        w_I,
        norm_w_I,
        beta,
        norm_w_S,
        aspect_ratio,
        delta_S,
        q1,
        q2,
        q3,
        Mp1,
        Mp2,
        Mp3,
    ) = params
    (
        r_min_err,
        I_rmin_err,
        delta_I_err,
        w_I_err,
        norm_w_I_err,
        beta_err,
        norm_w_S_err,
        aspect_ratio_err,
        delta_S_err,
        q1_err,
        q2_err,
        q3_err,
        Mp1_err,
        Mp2_err,
        Mp3_err,
    ) = param_errs

    keck_Mp = 1.96  # M_j
    keck_q = keck_Mp / 1047 / target_info.stellar_mass
    keck_alpha = keck_q**2 / ((delta_S - 1) * (aspect_ratio) ** 5) * 0.043

    print(f"Qphi gap r_min: {r_min} ± {r_min_err} au, {I_rmin * 100} ± {I_rmin_err * 100} %")

    print(f"Assuming Qphi disk β: {beta} ± {beta_err}")
    print(f"Qphi disk gap width: {w_I} ± {w_I_err} au, ΔI: {norm_w_I} ± {norm_w_I_err}")
    print(f"Qphi disk h/r: {aspect_ratio} ± {aspect_ratio_err}")
    print(
        f"Qphi disk density width: {norm_w_S * r_min} ± {(norm_w_S * r_min) * np.hypot(norm_w_S_err/norm_w_S, r_min_err / r_min)} au, ΔI: {norm_w_S} ± {norm_w_S_err}"
    )
    print(
        f"Qphi disk gap depth: {delta_I} ± {delta_I_err}, density depth: {delta_S} ± {delta_S_err}"
    )

    print("Using ⍺: [1e-4, 1e-3, 1e-2]")
    print(f"Qphi mass ratios: [{q1} ± {q1_err}, {q2} ± {q2_err}, {q3} ± {q3_err}]")
    print(f"Qphi planet masses: [{Mp1} ± {Mp1_err}, {Mp2} ± {Mp2_err}, {Mp3} ± {Mp3_err}] M_J")
    print(f"Qphi viscosity for {keck_Mp} MJ: {keck_alpha:.1e}")

    axes[0].axvline(r_min, c="0.3", ls="--", lw=1, alpha=0.7)

    axes[0].fill_betweenx(
        axes[0].get_ylim(),
        r_min - norm_w_S * r_min / 2,
        r_min + norm_w_S * r_min / 2,
        c="0.3",
        lw=0,
        alpha=0.1,
    )

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
        r_min,
        1.02,
        "D2",
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
