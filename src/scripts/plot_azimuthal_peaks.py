import proplot as pro
import paths
import numpy as np
import pandas as pd
from target_info import target_info
from utils_ephemerides import keplerian_warp
from utils_errorprop import relative_deviation
from utils_organization import folders, pxscales, label_from_folder, time_from_folder
from utils_plots import setup_rc
from astropy.io import fits

if __name__ == "__main__":
    setup_rc()

    ## Plot and save
    width = 3.31314
    aspect_ratio = 1 / (3 * 1.61803)
    height = width * aspect_ratio
    fig, axes = pro.subplots(
        nrows=8, width=f"{width}in", refheight=f"{height}in", hspace=0.7
    )

    labels = [label_from_folder(f) for f in folders]

    # colors = [f"C{i}" for i in range(len(folders))]
    for folder_idx, folder in enumerate(folders):

    # load data
        polar_cube = fits.getdata(
            paths.data
            / folder
            / f"{folder}_HD169142_Qphi_polar.fits"
        )

        rin = np.floor(15 / target_info.dist_pc / pxscales[folder]).astype(int)
        rout = np.ceil(35 / target_info.dist_pc / pxscales[folder]).astype(int)

        rs = np.arange(polar_cube.shape[0])

        mask = (rs >= rin) & (rs <= rout)
        r_peaks = np.nanargmax(polar_cube[mask, :], axis=0) + rin
        r_peaks_au = r_peaks * target_info.dist_pc * pxscales[folder]
        azimuth_deg = np.arange(polar_cube.shape[1]) * 5
        
        axes[folder_idx].plot(
            azimuth_deg,
            r_peaks_au,
            c="C0",
        )        
        labels = label_from_folder(folder).split()
        axes[folder_idx].text(
            0.01, 0.95, labels[0], transform="axes", c="0.1 ", ha="left", va="top", fontweight="bold"
        )
        axes[folder_idx].text(
            0.99, 0.95, "\n".join(labels[1:]), transform="axes", c="0.1 ", ha="right", va="top", fontweight="bold"
        )

    # for ax in axes:
        # ax.axhline(0, c="0.3", lw=1, zorder=0)

        # for offset in (90, 270):
        #     ax.axvline(offset + target_info.pos_angle, c="0.1", lw=1)

    axes.format(
        ylim=(15, 30),
        xlabel="Azimuth (° East of North)",
        ylabel=r"Separation (au)",
        xlocator=90,
    )

    fig.savefig(
        paths.figures / "HD169142_azimuthal_peaks.pdf",
        bbox_inches="tight",
    )

#########
    reference_time = time_from_folder("20180715_ZIMPOL")

    fig, axes = pro.subplots(
        nrows=8, width=f"{width}in", refheight=f"{height}in", hspace=0.7
    )

    labels = [label_from_folder(f) for f in folders]

    # colors = [f"C{i}" for i in range(len(folders))]
    for folder_idx, folder in enumerate(folders):

    # load data
        polar_cube = fits.getdata(
            paths.data
            / folder
            / f"{folder}_HD169142_Qphi_polar.fits"
        )

        rin = np.floor(15 / target_info.dist_pc / pxscales[folder]).astype(int)
        rout = np.ceil(35 / target_info.dist_pc / pxscales[folder]).astype(int)

        rs = np.arange(polar_cube.shape[0])

        mask = (rs >= rin) & (rs <= rout)
        rs_au = rs[mask] * target_info.dist_pc * pxscales[folder]
        polar_cube_rolled = keplerian_warp(polar_cube[mask, :], rs_au, time_from_folder(folder), reference_time)

        r_peaks = np.nanargmax(polar_cube_rolled, axis=0) + rin
        r_peaks_au = r_peaks * target_info.dist_pc * pxscales[folder]
        azimuth_deg = np.arange(polar_cube.shape[1]) * 5
        
        axes[folder_idx].plot(
            azimuth_deg,
            r_peaks_au,
            c="C0",
        )        
        labels = label_from_folder(folder).split()
        axes[folder_idx].text(
            0.01, 0.95, labels[0], transform="axes", c="0.1 ", ha="left", va="top", fontweight="bold"
        )
        axes[folder_idx].text(
            0.99, 0.95, "\n".join(labels[1:]), transform="axes", c="0.1 ", ha="right", va="top", fontweight="bold"
        )

    # for ax in axes:
        # ax.axhline(0, c="0.3", lw=1, zorder=0)

        # for offset in (90, 270):
        #     ax.axvline(offset + target_info.pos_angle, c="0.1", lw=1)

    axes.format(
        ylim=(15, 30),
        xlabel="Azimuth (° East of North)",
        ylabel=r"Separation (au)",
        xlocator=90,
    )

    fig.savefig(
        paths.figures / "HD169142_azimuthal_peaks_warped.pdf",
        bbox_inches="tight",
    )