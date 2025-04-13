import cv2
import numpy as np
import paths
import proplot as pro
import tqdm
from astropy.io import fits
from target_info import target_info
from utils_ephemerides import keplerian_warp
from utils_organization import folders, pxscales, time_from_folder
from utils_plots import setup_rc

if __name__ == "__main__":
    setup_rc()

    common_rs = np.linspace(15, 35, 100)
    common_thetas = np.arange(0, 360 // 5)
    thetas_grid, rs_grid = np.meshgrid(common_thetas, common_rs)

    alma_folder = "20170918_ALMA_1.3mm"
    alma_data = fits.getdata(paths.data / alma_folder / f"{alma_folder}_HD169142_Qphi_polar.fits")
    rs = np.arange(alma_data.shape[0])
    rin = np.floor(15 / target_info.dist_pc / pxscales[alma_folder])
    rout = np.ceil(35 / target_info.dist_pc / pxscales[alma_folder])
    mask = (rs >= rin) & (rs <= rout)
    alma_data = alma_data[mask, :]
    rs_au = rs[mask] * target_info.dist_pc * pxscales[alma_folder]

    rs_grid_norm = (rs_grid - rs_au.min()) / (target_info.dist_pc * pxscales[alma_folder])
    alma_data = cv2.remap(
        alma_data.astype("f4"),
        thetas_grid.astype("f4"),
        rs_grid_norm.astype("f4"),
        cv2.INTER_LANCZOS4,
    )

    alma_time = time_from_folder(alma_folder)

    ## Plot and save
    width = 3.31314
    aspect_ratio = 1 / 1.6
    height = width * aspect_ratio
    fig, axes = pro.subplots(width=f"{width}in", height=f"{height}in")

    images = []
    for _i, folder in enumerate(tqdm.tqdm(folders)):
        # load data
        polar_frame = fits.getdata(paths.data / folder / f"{folder}_HD169142_Qphi_polar.fits")

        rin = np.floor(15 / (target_info.dist_pc * pxscales[folder]))
        rout = np.ceil(35 / (target_info.dist_pc * pxscales[folder]))

        rs = np.arange(polar_frame.shape[0])
        mask = (rs >= rin) & (rs <= rout)

        rs_au = rs[mask] * target_info.dist_pc * pxscales[folder]

        this_time = time_from_folder(folder)
        polar_frame_warped = keplerian_warp(polar_frame[mask, :], rs_au, this_time, alma_time)

        rs_grid_norm = (rs_grid - rs_au.min()) / (target_info.dist_pc * pxscales[folder])
        polar_frame_regrid = cv2.remap(
            polar_frame_warped.astype("f4"),
            thetas_grid.astype("f4"),
            rs_grid_norm.astype("f4"),
            cv2.INTER_LANCZOS4,
        )

        images.append(polar_frame_regrid / np.nanmedian(polar_frame_regrid))

    data = np.nanmean(images, axis=0)
    # PDI images
    levels = np.nanpercentile(data, [80, 90, 97])
    im = axes[0].contour(thetas_grid * 5, rs_grid, data, c="C0", levels=levels, zorder=10)

    levels = np.nanpercentile(alma_data, [63, 75, 90])
    im = axes[0].contour(thetas_grid * 5, rs_grid, alma_data, c="C3", levels=levels, zorder=5)

    axes[0].text(
        0.98,
        0.02,
        r"Mean $Q_\phi \times r^2$",
        c="C0",
        fontweight="bold",
        transform="axes",
        ha="right",
        va="bottom",
    )
    axes[0].text(
        0.98,
        0.98,
        "ALMA (1.3mm)",
        c="C3",
        fontweight="bold",
        transform="axes",
        ha="right",
        va="top",
    )

    ## sup title
    axes.format(aspect="auto", xlabel="Angle E of N (°)", ylabel="Separation (au)", xlocator=90)

    fig.savefig(
        paths.figures / "HD169142_polar_median_ALMA_contours.pdf", bbox_inches="tight", dpi=300
    )
