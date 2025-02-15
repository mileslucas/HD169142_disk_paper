import proplot as pro
import numpy as np
import paths
from astropy.io import fits
import tqdm
from target_info import target_info
from utils_ephemerides import blob_c_position, blob_d_position
from utils_organization import folders, pxscales, label_from_folder, time_from_folder
from utils_plots import setup_rc

if __name__ == "__main__":
    setup_rc()
    pro.rc["axes.grid"] = False
    pro.rc["axes.facecolor"] = "w"

    ## Plot and save
    width = 3.31314
    aspect_ratio = 1 / (3 * 1.61803)
    height = width * aspect_ratio
    fig, axes = pro.subplots(
        nrows=8, width=f"{width}in", refheight=f"{height}in", hspace=0.5
    )

    def format_date(date):
        return f"{date[:4]}/{date[4:6]}"

    for i, folder in enumerate(tqdm.tqdm(folders)):
    # load data
        with fits.open(
            paths.data
            / folder
            / f"{folder}_HD169142_Qphi_polar.fits"
        ) as hdul:
            polar_cube = hdul[0].data

        rin = np.floor(15 / target_info.dist_pc / pxscales[folder]).astype(int)
        rout = np.ceil(45 / target_info.dist_pc / pxscales[folder]).astype(int)
        
        rs = np.arange(polar_cube.shape[0])

        mask = (rs >= rin) & (rs <= rout)
        ext = (0, 360, rin * pxscales[folder] * target_info.dist_pc, rout * pxscales[folder] * target_info.dist_pc)

        timestamp = time_from_folder(folder)
        c_a, c_th = blob_c_position(timestamp)
        d_a, d_th = blob_d_position(timestamp)

        axes[i].scatter(c_th, c_a, marker="^", ms=30, c="0.1", lw=1)
        axes[i].scatter(d_th, d_a, marker="v", ms=30, c="0.1", lw=1)

        # PDI images
        image = polar_cube[mask, :]
        image_mean = np.nanmedian(image, axis=1, keepdims=True)
        norm_image = image - image_mean

        norm = pro.DivergingNorm()
        im = axes[i].imshow(norm_image, extent=ext, cmap="div", norm=norm)
        # axes[0].colorbar(im)
        labels = label_from_folder(folder).split()
        axes[i].text(
            0.01, 0.95, labels[0], transform="axes", c="0.1 ", ha="left", va="top", fontweight="bold"
        )
        axes[i].text(
            0.99, 0.95, "\n".join(labels[1:]), transform="axes", c="0.1 ", ha="right", va="top", fontweight="bold"
        )

    for ax in axes:
        for offset in (90, 270):
            ax.axvline(offset + target_info.pos_angle, c="0.1", lw=1)

    ## sup title
    axes.format(
        aspect="auto",
        xlabel="Angle E of N (°)",
        ylabel="Separation (au)",
        xlocator=90,
    )

    axes[:-1].format(xtickloc="none")

    fig.savefig(
        paths.figures / "HD169142_polar_collapsed_inner_subbed.pdf", bbox_inches="tight", dpi=300
    )
