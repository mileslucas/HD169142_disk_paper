import proplot as pro
import numpy as np
import paths
from astropy.io import fits
import tqdm
from astropy.visualization import simple_norm

from target_info import target_info
from utils_ephemerides import keplerian_warp
from utils_organization import label_from_folder, time_from_folder, folders, pxscales
from utils_plots import setup_rc

if __name__ == "__main__":
    setup_rc()
    pro.rc["axes.grid"] = False
    pro.rc["axes.facecolor"] = "k"

    timestamps = list(map(time_from_folder, folders))
    ## Plot and save
    height = 3.31314
    width = 2.3 * height
    fig, axes = pro.subplots(
        ncols=8, height=f"{height}in", width=f"{width}in", wspace=0.5
    )

    for i, folder in enumerate(tqdm.tqdm(folders)):
    # load data
        with fits.open(
            paths.data
            / folder
            / f"{folder}_HD169142_Qphi_polar.fits"
        ) as hdul:
            polar_cube = hdul[0].data

        rin = np.floor(48 / target_info.dist_pc / pxscales[folder]).astype(int)
        rout = np.ceil(110 / target_info.dist_pc / pxscales[folder]).astype(int)

        rs = np.arange(polar_cube.shape[0])

        mask = (rs >= rin) & (rs <= rout)
        ext = (rin * pxscales[folder] * target_info.dist_pc, rout * pxscales[folder] * target_info.dist_pc, 360, 0)

        rs_au = rs[mask] * target_info.dist_pc * pxscales[folder]
        polar_cube_rolled = keplerian_warp(polar_cube[mask, :], rs_au, timestamps[i], timestamps[4])


        # PDI images
        data = np.flipud(polar_cube_rolled.T)
        norm = simple_norm(data, vmin=0, stretch="sinh", sinh_a=0.5)
        im = axes[i].imshow(data, extent=ext, norm=norm, vmin=norm.vmin, vmax=norm.vmax, cmap=pro.rc["cmap"])
        # axes[0].colorbar(im)
        labels = label_from_folder(folder).split()
        axes[i].text(
            0.95, 0.99, labels[0], transform="axes", c="white", ha="right", va="top", fontweight="bold", rotation=-90
        )
        axes[i].text(
            0.95, 0.01, " ".join(labels[1:]), transform="axes", c="white", ha="right", va="bottom", fontweight="bold", rotation=-90
        )

    ## sup title
    axes.format(
        aspect="auto",
        ylabel="Angle E of N (°)",
        xlabel="Separation (au)",
        ylocator=90,
    )
    axes[1:].format(ytickloc="none")


    fig.savefig(
        paths.figures / "HD169142_polar_collapsed_outer_rolled_presentation.pdf", bbox_inches="tight"
    )
