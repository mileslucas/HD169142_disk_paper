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
from utils_errorprop import bootstrap_peak

if __name__ == "__main__": 
    setup_rc()

    alma_folder = "20170918_ALMA_1.3mm"
    alma_table = pd.read_csv(paths.data / alma_folder / f"{alma_folder}_HD169142_radial_profiles.csv")
    alma_radii = alma_table["radius(au)"].values
    alma_curve = alma_table["I"].values / alma_table["I"].max()
    alma_err = alma_table["I_err"].values / alma_table["I"].max()
    alma_time = time_from_folder(alma_folder)
    
    ## Plot and save
    width = 3.31314
    aspect_ratio = 1/1.6
    height = width * aspect_ratio
    fig, axes = pro.subplots(
        nrows=1, width=f"{width}in", height=f"{height}in", hspace=0.25
    )

    common_rs = np.linspace(0, alma_radii.max(), 2 * len(alma_radii))
    curves = []
    errs = []
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



    axes[0].plot(common_rs, mean_curve, shadedata=err_curve, c="C0", label=r"Mean $Q_\phi \times r^2$", zorder=10)
    axes[0].plot(alma_radii, alma_curve, shadedata=alma_err, c="C3", label="ALMA (1.3mm)", zorder=9)



    qphi_peak, qphi_peak_err = bootstrap_peak(common_rs, mean_curve, err_curve)
    print(f"Qphi peak: {qphi_peak} ± {qphi_peak_err} au")
    
    alma_peak, alma_peak_err = bootstrap_peak(alma_radii, alma_curve, alma_err)
    print(f"ALMA peak: {alma_peak} ± {alma_peak_err} au")

    with open(paths.data / "radial_profile_peaks.csv", "w") as fh:
        fh.write(f"Qphi,{qphi_peak},{qphi_peak_err}\n")
        fh.write(f"ALMA,{alma_peak},{alma_peak_err}\n")



    axes[0].axvline(qphi_peak, c="C0", zorder=0, lw=1, alpha=0.7)
    axes[0].axvline(alma_peak, c="C3", zorder=0, lw=1, alpha=0.7)

    axes[0].legend(ncols=1)

    ## sup title
    axes.format(
        ylim=(-0.1, None),
        xlim=(0, 115),
        xlabel="Separation (au)",
        ylabel="Normalized profile",
    ) 


    fig.savefig(
        paths.figures / "HD169142_radial_profiles_Qphi_combined_ALMA.pdf", bbox_inches="tight"
    )


