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
    alma_prof = np.nanmean(alma_polar_cube[mask, :], axis=0)
    alma_norm_prof = alma_prof / np.mean(alma_prof) - 1

    ## Plot and save
    fig, axes = pro.subplots(width="3.33in", refheight="1.5in")

    profiles = []

    for i, folder in enumerate(tqdm.tqdm(folders)):

    # load data
        with fits.open(
            paths.data
            / folder
            / f"{folder}_HD169142_Qphi_polar.fits"
        ) as hdul:
            polar_cube = hdul[0].data


        rin = np.floor(15 / target_info.dist_pc / pxscales[folder]).astype(int)
        rout = np.ceil(35 / target_info.dist_pc / pxscales[folder]).astype(int)

        rs = np.arange(polar_cube.shape[0])

        mask = (rs >= rin) & (rs <= rout)
        ext = (0, 360, rin * pxscales[folder] * target_info.dist_pc, rout * pxscales[folder] * target_info.dist_pc)
        rs_au = rs[mask] * target_info.dist_pc * pxscales[folder]
        polar_cube_rolled = keplerian_warp(polar_cube[mask, :], rs_au, timestamps[i], alma_timestamp)
        profile = np.nanmean(polar_cube_rolled, axis=0)
        norm_profile = profile / profile.mean() - 1
        profiles.append(norm_profile)
        # PDI images

    theta = np.arange(0, 360, 5)

    mean_prof =  np.nanmean(profiles, axis=0)
    norm_prof = mean_prof# / np.mean(mean_prof) - 1
    axes[0].plot(theta, norm_prof * 100, c="C0", zorder=100)

    axes[0].plot(theta, alma_norm_prof * 100, c="C3", zorder=100)
    axes[0].text(
        0.15, 0.98,
        r"Mean $Q_\phi \times r^2$",
        c="C0",
        fontweight="bold",
        transform="axes",
        ha="left",
        va="top",
    )
    axes[0].text(
        0.15, 0.9,
        "ALMA (1.3mm)",
        c="C3",
        fontweight="bold",
        transform="axes",
        ha="left",
        va="top",
    )

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


    