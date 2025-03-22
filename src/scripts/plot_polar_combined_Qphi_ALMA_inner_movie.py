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
import cv2
import os

from utils_organization import folders, pxscales, time_from_folder
from utils_ephemerides import keplerian_warp
from utils_plots import setup_rc

if __name__ == "__main__":
    setup_rc()
    pro.rc["axes.grid"] = False
    pro.rc["axes.facecolor"] = "k"

    alma_folder = "20170918_ALMA_1.3mm"
    alma_data = fits.getdata(paths.data / alma_folder / f"{alma_folder}_HD169142_Qphi_polar.fits")
    rs = np.arange(alma_data.shape[0])
    rin = np.floor(15 / target_info.dist_pc / pxscales[alma_folder]).astype(int)
    rout = np.ceil(35 / target_info.dist_pc / pxscales[alma_folder]).astype(int)
    mask = (rs >= rin) & (rs <= rout)
    alma_data = alma_data[mask, :]
    alma_curve = np.nanmean(alma_data, axis=0)
    alma_curve = alma_curve / alma_curve.mean() - 1
    alma_err = np.nanstd(alma_data, axis=0)

    alma_time = time_from_folder(alma_folder)

    ## Plot and save
    width = 3.31314
    aspect_ratio = 1/2
    height = width * aspect_ratio
    fig, axes = pro.subplots(width=f"{width}in", height=f"{height}in")

    common_rs = np.linspace(15, 35, alma_data.shape[0])
    common_thetas = np.arange(0, 360//5)
    thetas_grid, rs_grid = np.meshgrid(common_thetas, common_rs)
    images= []
    for i, folder in enumerate(tqdm.tqdm(folders)):

    # load data
        with fits.open(
            paths.data
            / folder
            / f"{folder}_HD169142_Qphi_polar.fits"
        ) as hdul:
            polar_cube = hdul[0].data


        rin = np.floor(15 / (target_info.dist_pc * pxscales[folder])).astype(int)
        rout = np.ceil(35 / (target_info.dist_pc * pxscales[folder])).astype(int)

        rs = np.arange(polar_cube.shape[0])

        mask = (rs >= rin) & (rs <= rout)
        ext = (0, 360, rin * pxscales[folder] * target_info.dist_pc, rout * pxscales[folder] * target_info.dist_pc)
        rs_au = rs[mask] * target_info.dist_pc * pxscales[folder]

        this_time = time_from_folder(folder)
        polar_cube_warped = keplerian_warp(polar_cube[mask, :], rs_au, this_time, alma_time)

        rs_grid_norm = (rs_grid - rs_au.min()) / (target_info.dist_pc * pxscales[folder])
        data = cv2.remap(polar_cube_warped, thetas_grid.astype("f4"), rs_grid_norm.astype("f4"), cv2.INTER_LANCZOS4)

        images.append(data / np.nanmedian(data))

    # axes[0].axhline(20, c="w", ls=":", lw=0.7, alpha=0.8)
    # axes[0].text(0.99, 0.15, r"H$_2$O snowline", c="w", alpha=0.9, fontsize=6, transform="axes", ha="right", va="center")
    # axes[1].axhline(20, c="0.9", lw=0.5, alpha=0.8)

    data = np.nanmean(images, axis=0)
    # PDI images
    norm = simple_norm(data, vmin=0, stretch="sinh", sinh_a=0.5)
    im = axes[0].imshow(data, extent=ext, norm=norm, vmin=norm.vmin, vmax=norm.vmax)

    label = axes[0].text(
        0.01, 0.95, r"Mean $Q_\phi \times r^2$", c="white", ha="left", va="top", transform="axes"
    )

    ## sup title
    axes.format(
        aspect="auto",
        xlabel="Angle E of N (°)",
        ylabel="Separation (au)",
        xlocator=90,
    )

    path_1 = paths.figures / "HD169142_polar_median_Qphi_inner.pdf"
    fig.savefig(path_1, bbox_inches="tight")


    norm = simple_norm(alma_data, vmin=0)#, stretch="sinh", sinh_a=0.5)
    im = axes[0].imshow(alma_data, extent=ext, norm=norm, vmin=norm.vmin, vmax=norm.vmax, cmap="inferno")

    label.set_text("ALMA (1.3 mm)")
    label.set_ha("right")
    label.set_position((0.99, 0.95))
    
    axes.format(
        aspect="auto",
        xlabel="Angle E of N (°)",
        ylabel="Separation (au)",
        xlocator=90,
    )

    path_2 = paths.figures / "HD169142_polar_median_ALMA_inner.pdf"
    fig.savefig(path_2, bbox_inches="tight")

    # call imagemagick

    gif_duration = 10 # seconds
    gif_fps = 12
    gif_frames = gif_duration * gif_fps

    gif_name = paths.figures / "HD169142_polar_median_inner_transition_movie.gif"
    dpi = pro.rc["figure.dpi"]
    cmd = f"magick convert -delay {100 // gif_fps} -loop 0 -density {dpi} -dispose previous {path_1} {path_2} {path_1} -morph {gif_frames // 2} {gif_name}"
    os.system(cmd)
