import proplot as pro
import paths
import numpy as np
import pandas as pd
from target_info import target_info
from utils_ephemerides import keplerian_warp
from utils_errorprop import bootstrap_argmax_and_max
from utils_organization import folders, pxscales, label_from_folder, time_from_folder
from utils_plots import setup_rc
from astropy.io import fits
import tqdm

if __name__ == "__main__":
    setup_rc()

    ## Plot and save
    width = 3.31314
    aspect_ratio = 1.61803
    height = width * aspect_ratio
    fig, axes = pro.subplots(
        nrows=8, ncols=2, width=f"{width}in", height=f"{height}in", wspace=0.25, hspace=0.5,
    )

    labels = [label_from_folder(f) for f in folders]
    reference_time = time_from_folder("20180715_ZIMPOL")

    # colors = [f"C{i}" for i in range(len(folders))]
    for folder_idx, folder in enumerate(tqdm.tqdm(folders)):
        Qphi_polar = fits.getdata(paths.data / folder / f"{folder}_HD169142_Qphi_polar.fits")
        Uphi_polar = fits.getdata(paths.data / folder / f"{folder}_HD169142_Uphi_polar.fits")
        rin = np.floor(15 / target_info.dist_pc / pxscales[folder]).astype(int)
        rout = np.ceil(35 / target_info.dist_pc / pxscales[folder]).astype(int)

        rs = np.arange(Qphi_polar.shape[0])
        mask = (rs >= rin) & (rs <= rout)
        rs_au = rs[mask] * target_info.dist_pc * pxscales[folder]

        peaks = []
        peak_errs = []
        for az_idx in range(Qphi_polar.shape[1]):
            Qphi_slice = Qphi_polar[mask, az_idx]
            Uphi_slice = Uphi_polar[mask, az_idx]
            peak, peakerr, _, _ = bootstrap_argmax_and_max(rs[mask], Qphi_slice, Uphi_slice, 1000)
            peaks.append(peak)
            peak_errs.append(peakerr)

        r_peaks_au = np.array(peaks) * target_info.dist_pc * pxscales[folder]
        r_peak_errs_au = np.array(peak_errs) * target_info.dist_pc * pxscales[folder]
        azimuth_deg = np.arange(Qphi_polar.shape[1]) * 5

        axes[folder_idx, 0].plot(
            azimuth_deg,
            r_peaks_au,
            shadedata=r_peak_errs_au,
            c="C1",
        )

        Qphi_polar_warped = keplerian_warp(Qphi_polar[mask, :], rs_au, time_from_folder(folder), reference_time)
        Uphi_polar_warped = keplerian_warp(Uphi_polar[mask, :], rs_au, time_from_folder(folder), reference_time)

        peaks = []
        peak_errs = []
        for az_idx in range(Qphi_polar_warped.shape[1]):
            Qphi_slice = Qphi_polar_warped[:, az_idx]
            Uphi_slice = Uphi_polar_warped[:, az_idx]
            peak, peakerr, _, _ = bootstrap_argmax_and_max(rs[mask], Qphi_slice, Uphi_slice, 1000)
            peaks.append(peak)
            peak_errs.append(peakerr)

        r_peaks_au = np.array(peaks) * target_info.dist_pc * pxscales[folder]
        r_peak_errs_au = np.array(peak_errs) * target_info.dist_pc * pxscales[folder]
        azimuth_deg = np.arange(Qphi_polar.shape[1]) * 5

        axes[folder_idx, 1].plot(
            azimuth_deg,
            r_peaks_au,
            shadedata=r_peak_errs_au,
            c="C1",
        )

        labels = label_from_folder(folder).split()
        axes[folder_idx, 0].text(
            0.01, 0.95, labels[0], transform="axes", c="0.1 ", ha="left", va="top", fontweight="bold", fontsize=7
        )
        axes[folder_idx, 1].text(
            0.99, 0.95, " ".join(labels[1:]), transform="axes", c="0.1 ", ha="right", va="top", fontweight="bold", fontsize=7
        )

    for ax in axes:
        ax.axhline(20.9, c="0.3", lw=1, zorder=0) # radial profile peak

    axes[0, 0].format(title="Original")
    axes[0, 1].format(title="Keplerian warped")

    axes.format(
        xlabel="Azimuth (° East of North)",
        ylabel=r"Separation (au)",
        xlocator=90,
    )

    fig.savefig(
        paths.figures / "HD169142_azimuthal_peaks.pdf",
        bbox_inches="tight",
    )

