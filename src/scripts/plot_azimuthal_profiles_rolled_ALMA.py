import proplot as pro
import numpy as np
import paths
from astropy.io import fits
from skimage.transform import warp_polar
from astropy.convolution import convolve, kernels
import tqdm
from astropy.visualization import simple_norm
from target_info import target_info
from astropy import time
from utils_plots import setup_rc
from utils_organization import folders, pxscales, time_from_folder
from utils_ephemerides import keplerian_warp
from utils_errorprop import relative_deviation

if __name__ == "__main__": 
    setup_rc()

    timestamps = list(map(time_from_folder, folders))
  

    alma_folder = "20170918_ALMA_1.3mm"
    alma_polar_cube = fits.getdata(paths.data / alma_folder / f"{alma_folder}_HD169142_Qphi_polar.fits")

    rin = np.floor(15 / target_info.dist_pc / pxscales[alma_folder]).astype(int)
    rout = np.ceil(35 / target_info.dist_pc / pxscales[alma_folder]).astype(int)

    rs = np.arange(alma_polar_cube.shape[0])
    mask = (rs >= rin) & (rs <= rout)

    alma_timestamp = time_from_folder(alma_folder)
    alma_prof = np.mean(alma_polar_cube[mask, :], axis=0)
    alma_prof_std = np.std(alma_polar_cube[mask, :], axis=0) / np.sqrt((rout - rin))
    alma_prof_error = np.hypot(14.7e-3 / np.sqrt((rout - rin)), alma_prof_std)
    alma_norm_prof, alma_norm_err = relative_deviation(alma_prof, alma_prof_error)

    ## Plot and save
    fig, axes = pro.subplots(width="3.33in", refheight="1.5in")

    profiles = []
    errs = []

    for i, folder in enumerate(tqdm.tqdm(folders)):

        # load data
        Qphi_polar = fits.getdata(paths.data / folder / f"{folder}_HD169142_Qphi_polar.fits")
        Uphi_polar = fits.getdata(paths.data / folder / f"{folder}_HD169142_Uphi_polar.fits")


        rin = np.floor(15 / target_info.dist_pc / pxscales[folder]).astype(int)
        rout = np.ceil(35 / target_info.dist_pc / pxscales[folder]).astype(int)

        rs = np.arange(Qphi_polar.shape[0])

        mask = (rs >= rin) & (rs <= rout)
        ext = (0, 360, rin * pxscales[folder] * target_info.dist_pc, rout * pxscales[folder] * target_info.dist_pc)
        rs_au = rs[mask] * target_info.dist_pc * pxscales[folder]
        Qphi_polar_rolled = keplerian_warp(Qphi_polar[mask, :], rs_au, timestamps[i], alma_timestamp)
        Uphi_polar_rolled = keplerian_warp(Uphi_polar[mask, :], rs_au, timestamps[i], alma_timestamp)
        profile = np.mean(Qphi_polar_rolled, axis=0)
        std = np.std(Qphi_polar_rolled, axis=0)
        stderr = std / np.sqrt(Qphi_polar_rolled.shape[0])
        rmserr = np.sqrt(np.sum(Uphi_polar_rolled**2, axis=0)) / Uphi_polar_rolled.shape[0]
        err = np.hypot(stderr, rmserr)
        norm_prof, norm_err = relative_deviation(profile, err)
        profiles.append(norm_prof)
        errs.append(norm_err)
        # PDI images

    theta = np.arange(0, 360, 5)

    mean_prof =  np.mean(profiles, axis=0)
    stderr_prof = np.std(profiles, axis=0) / np.sqrt(len(profiles))
    rms_prof = np.sqrt(np.sum(np.power(errs, 2), axis=0)) / len(errs)
    norm_prof = mean_prof
    norm_err = np.hypot(stderr_prof, rms_prof)
    axes[0].plot(theta, norm_prof * 100, shadedata=norm_err * 100, c="C0", zorder=100, label=r"Mean $Q_\phi \times r^2$")

    axes[0].plot(theta, alma_norm_prof * 100, shadedata=alma_norm_err * 100, c="C3", zorder=90, label="ALMA (1.3mm)")
    axes[0].legend(ncols=1)

    for ax in axes:
        ax.axhline(0, c="0.3", zorder=0, lw=1)

    ## sup title
    axes.format(
        xlabel="Angle E of N (°)",
        xlocator=90,
        yformatter="percent"
    )

    # axes[:-1].format(xtickloc="none")

    fig.savefig(
        paths.figures / "HD169142_azimuth_profile_inner_rolled_ALMA.pdf", bbox_inches="tight", dpi=300
    )


    