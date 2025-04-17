import cv2
import numpy as np
import paths
import proplot as pro
import tqdm
from astropy.io import fits
from astropy.visualization import simple_norm
from matplotlib import patches
from target_info import target_info
from utils_ephemerides import keplerian_warp
from utils_organization import folders, pxscales, time_from_folder
from utils_plots import setup_rc

if __name__ == "__main__":
    setup_rc()
    pro.rc["axes.grid"] = False
    pro.rc["axes.facecolor"] = "k"

    alma_folder = "20170918_ALMA_1.3mm"
    alma_data = fits.getdata(paths.data / alma_folder / f"{alma_folder}_HD169142_Qphi_polar.fits")

    alma_time = time_from_folder(alma_folder)

    ## Plot and save
    width = 3.31314
    aspect_ratio = 1 / 2
    height = width * aspect_ratio
    fig, axes = pro.subplots(width=f"{width}in", height=f"{height}in")

    common_rs = np.linspace(15, 35, alma_data.shape[0])
    common_thetas = np.arange(0, 360 // 5)
    thetas_grid, rs_grid = np.meshgrid(common_thetas, common_rs)
    images = []
    for _i, folder in enumerate(tqdm.tqdm(folders)):
        # load data
        with fits.open(paths.data / folder / f"{folder}_HD169142_Qphi_polar.fits") as hdul:
            polar_cube = hdul[0].data

        rin = np.floor(15 / (target_info.dist_pc * pxscales[folder]))
        rout = np.ceil(35 / (target_info.dist_pc * pxscales[folder]))

        rs = np.arange(polar_cube.shape[0])

        mask = (rs >= rin) & (rs <= rout)
        rs_au = rs[mask] * target_info.dist_pc * pxscales[folder]

        this_time = time_from_folder(folder)
        polar_cube_warped = keplerian_warp(polar_cube[mask, :], rs_au, this_time, alma_time)

        rs_grid_norm = (rs_grid - rs_au.min()) / (target_info.dist_pc * pxscales[folder])
        data = cv2.remap(
            polar_cube_warped,
            thetas_grid.astype("f4"),
            rs_grid_norm.astype("f4"),
            cv2.INTER_LANCZOS4,
        )

        images.append(data / np.nanmedian(data))

    data = np.nanmean(images, axis=0)
    ext = (0, 360, common_rs.min(), common_rs.max())
    # PDI images
    norm = simple_norm(data, vmin=0, stretch="sinh", sinh_a=0.5)
    im = axes[0].imshow(data, extent=ext, norm=norm, vmin=norm.vmin, vmax=norm.vmax)

    axes[0].text(
        0.01, 0.95, r"Mean $Q_\phi \times r^2$", c="white", ha="left", va="top", transform="axes"
    )

    ## sup title
    axes.format(aspect="auto", xlabel="Angle E of N (°)", ylabel="Separation (au)", xlocator=90)

    # Compute the display-to-data aspect correction factor
    radius_axes = 1 / 3

    inv = axes[0].transData.inverted()
    x0, y0 = inv.transform(axes[0].transAxes.transform((0, 0)))
    x1, y1 = inv.transform(
        axes[0].transAxes.transform((radius_axes / 5, radius_axes * aspect_ratio))
    )

    # Data-equivalent radius
    radius_x_data = x1 - x0
    radius_y_data = y1 - y0

    # Now draw a circle in data coords with those scaled radii
    patch_kwargs = dict(
        width=2 * radius_x_data, height=2 * radius_y_data, color="#16ff68", linewidth=1, fill=False
    )
    text_kwargs = dict(color="#16ff68", ha="center", va="bottom", fontweight="bold")

    circle = patches.Ellipse((10, 20.75), **patch_kwargs)
    axes[0].add_patch(circle)
    axes[0].text(10, 20.75 + 4, "C1", **text_kwargs)

    axes[0].text(45, 20.75 + 2, "S1", **text_kwargs)

    circle = patches.Ellipse((80, 20.75), **patch_kwargs)
    axes[0].add_patch(circle)
    axes[0].text(80, 20.75 + 4, "C2", **text_kwargs)

    axes[0].text(125, 20.75 + 2, "S2", **text_kwargs)

    # circle = patches.Ellipse((180, 20.75), **patch_kwargs)
    # axes[0].add_patch(circle)
    # axes[0].text(180, 20.75 + 4, "C3", **text_kwargs)

    patch_kwargs["width"] *= 2.15
    circle = patches.Ellipse((275, 22.5), **patch_kwargs)
    axes[0].add_patch(circle)
    axes[0].text(275, 22.5 + 4, "C3", **text_kwargs)

    fig.savefig(paths.figures / "HD169142_polar_median_Qphi_inner_labeled.pdf", bbox_inches="tight")
