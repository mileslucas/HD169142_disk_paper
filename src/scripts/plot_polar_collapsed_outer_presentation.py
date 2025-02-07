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
    height = 3.31314
    width = 2.3 * height
    fig, axes = pro.subplots(
        ncols=8, height=f"{height}in", width=f"{width}in", wspace=0.5
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

        rin = np.floor(48 / target_info.dist_pc / pxscales[folder]).astype(int)
        rout = np.ceil(110 / target_info.dist_pc / pxscales[folder]).astype(int)
        
        rs = np.arange(polar_cube.shape[0])

        mask = (rs >= rin) & (rs <= rout)
        ext = (rin * pxscales[folder] * target_info.dist_pc, rout * pxscales[folder] * target_info.dist_pc, 360, 0)


        # PDI images
        norm = simple_norm(np.flipud(polar_cube[mask, :].T), vmin=0, stretch="sinh", sinh_a=0.5)
        im = axes[i].imshow(np.flipud(polar_cube[mask, :].T), extent=ext, norm=norm, vmin=norm.vmin, vmax=norm.vmax, cmap=pro.rc["cmap"])
        # axes[0].colorbar(im)
        labels = label_from_folder(folder).split()
        axes[i].text(
            0.95, 0.99, labels[0], transform="axes", c="white", ha="right", va="top", fontweight="bold", rotation=-90
        )
        axes[i].text(
            0.95, 0.01, " ".join(labels[1:]), transform="axes", c="white", ha="right", va="bottom", fontweight="bold", rotation=-90
        )

        # axes[i].axhline(iwas[folder] / 1e3 * dist, c="w", alpha=0.4)

    # for ax in axes:
    #     norm_pa = np.mod(target_info.pos_angle - 90, 360)
    #     ax.axvline(norm_pa, lw=1, c="0.8")
    #     ax.axvline(norm_pa - 180, lw=1, c="0.8")


    ## sup title    
    axes.format(
        aspect="auto",
        ylabel="Angle E of N (°)",
        xlabel="Separation (au)",
        ylocator=90,
    )
    axes[1:].format(ytickloc="none")
    

    fig.savefig(
        paths.figures / "HD169142_polar_collapsed_outer_presentation.pdf", bbox_inches="tight"
    )
