import proplot as pro
import numpy as np
import paths
from astropy.io import fits
from skimage.transform import warp_polar
from astropy.convolution import convolve, kernels
import tqdm
from astropy.visualization import simple_norm
from target_info import target_info
from utils_organization import folders, pxscales, label_from_folder, time_from_folder
from utils_plots import setup_rc
from utils_ephemerides import keplerian_warp
from matplotlib import patches

if __name__ == "__main__":
    setup_rc()
    pro.rc["font.size"] = 6
    pro.rc["axes.grid"] = False
    pro.rc["axes.facecolor"] = "k"

    ## Plot and save
    width = 3.31314
    aspect_ratio = 1.61803
    height = width * aspect_ratio
    fig, axes = pro.subplots(
        nrows=8, ncols=2, width=f"{width}in", height=f"{height}in", hspace=0.5, wspace=0.5, spanx=False
    )
    timestamps = list(map(time_from_folder, folders))
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
        rs_au = rs[mask] * target_info.dist_pc * pxscales[folder]

        ext = (0, 360, rin * pxscales[folder] * target_info.dist_pc, rout * pxscales[folder] * target_info.dist_pc)

        # PDI images
        polar_cube_masked = polar_cube[mask, :]
        norm = simple_norm(polar_cube_masked, vmin=0, stretch="sinh", sinh_a=0.5)
        im = axes[i, 0].imshow(polar_cube_masked, extent=ext, norm=norm, vmin=norm.vmin, vmax=norm.vmax)
        # axes[0].colorbar(im)
        labels = label_from_folder(folder).split()
        axes[i, 0].text(
            0.01, 0.95, labels[0], transform="axes", c="white", ha="left", va="top",  fontweight="bold"
        )

        polar_cube_warped = keplerian_warp(polar_cube_masked, rs_au, timestamps[i], timestamps[4])

        im = axes[i, 1].imshow(polar_cube_warped, extent=ext, norm=norm, vmin=norm.vmin, vmax=norm.vmax)
        # axes[0].colorbar(im)
        labels = label_from_folder(folder).split()
        axes[i, 0].text(
            0.01, 0.95, labels[0], transform="axes", c="white", ha="left", va="top",  fontweight="bold"
        )

        axes[i, 1].text(
            0.99, 0.95, "\n".join(labels[1:]), transform="axes", c="white", ha="right", va="top",  fontweight="bold"
        )



    for ax in axes:
        for offset in (90,270):
            ax.vlines(offset + target_info.pos_angle, ax.get_ylim()[0], 30, c="0.9", lw=0.5, ls="--")

    axes[0, 0].format(title="Original")
    axes[0, 1].format(title="Keplerian warped")
    axes[:-1, :].format(xtickloc="none")
    axes[:, 1:].format(ytickloc="none")
    ## sup title
    axes.format(
        aspect="auto",
        xlabel="Angle E of N (°)",
        ylabel="",
        xlocator=90,
    )
    axes[:, 0].format(xlocator=[0, 90, 180, 270])

    fig.savefig(
        paths.figures / "HD169142_polar_Qphi_inner.pdf", bbox_inches="tight"
    )
