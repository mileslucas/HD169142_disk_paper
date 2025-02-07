import proplot as pro
import numpy as np
import paths
from astropy.io import fits
from skimage.transform import warp_polar
from astropy.convolution import convolve, kernels
import tqdm
from astropy.visualization import simple_norm
from target_info import target_info
from utils_organization import folders, pxscales, label_from_folder
from utils_plots import setup_rc

if __name__ == "__main__":
    setup_rc()
    pro.rc["axes.grid"] = False
    pro.rc["axes.facecolor"] = "k"

    ## Plot and save
    width = 3.31314
    aspect_ratio = 1 / (3 * 1.61803)
    height = width * aspect_ratio
    fig, axes = pro.subplots(
        nrows=8, width=f"{width}in", refheight=f"{height}in", hspace=0.5
    )

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


        # PDI images
        norm = simple_norm(polar_cube[mask, :], vmin=0, stretch="sinh", sinh_a=0.5)
        im = axes[i].imshow(polar_cube[mask, :], extent=ext, norm=norm, vmin=norm.vmin, vmax=norm.vmax, cmap=pro.rc["cmap"])
        # axes[0].colorbar(im)
        labels = label_from_folder(folder).split()
        axes[i].text(
            0.01, 0.95, labels[0], transform="axes", c="white", ha="left", va="top", fontweight="bold"
        )
        axes[i].text(
            0.99, 0.95, " ".join(labels[1:]), transform="axes", c="white", ha="right", va="top", fontweight="bold"
        )

    ## sup title
    axes.format(
        aspect="auto",
        xlabel="Angle E of N (°)",
        ylabel="Separation (au)",
        xlocator=90,
    )
    axes[:-1].format(xtickloc="none")

    fig.savefig(
        paths.figures / "HD169142_polar_collapsed_inner.pdf", bbox_inches="tight", dpi=300
    )
