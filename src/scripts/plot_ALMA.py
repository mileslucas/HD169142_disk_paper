import paths
import proplot as pro
from matplotlib import patches
import numpy as np
from utils_plots import setup_rc
from utils_organization import label_from_folder
from astropy.io import fits
from target_info import target_info

if __name__ == "__main__":
    setup_rc()
    pro.rc["grid"] = False

    alma_folder = "20170918_ALMA_1.3mm"
    alma_data, alma_hdr = fits.getdata(paths.data / alma_folder / "HD169142.selfcal.concat.GPU-UVMEM.centered_mJyBeam.fits", header=True)
    alma_pxscale = np.abs(alma_hdr["CDELT1"]) * 3.6e3 # arcsec / px
    alma_side_length = alma_data.shape[-1] * alma_pxscale / 2
    alma_ext = (alma_side_length, -alma_side_length, -alma_side_length, alma_side_length)

    width = 3.31314

    fig, ax = pro.subplots(width=f"{width}in")

    ax.imshow(alma_data, extent=alma_ext, cmap="magma", vmin=0)


    # text labels
    labels = label_from_folder(alma_folder).split()
    ax.text(
        0.03, 0.97, labels[0],
        transform="axes",
        c="w",
        fontweight="bold",
        ha="left",
        va="top"
    )
    ax.text(
        0.99, 0.97, " ".join(labels[1:]),
        transform="axes",
        c="w",
        fontweight="bold",
        ha="right",
        va="top"
    )
 # star position
    # ax.scatter(0, 0, marker="+", lw=0.7, markersize=20, c="white")
    # scale bar
    bar_width_arc = 0.3
    bar_width_height = bar_width_arc / 20
    bar_width_au = bar_width_arc * target_info.dist_pc
    rect = patches.Rectangle([0.75, -0.8 - bar_width_height/2], -bar_width_arc, bar_width_height, color="white")
    ax.add_patch(rect)

    ax.text(
        0.75 - bar_width_arc / 2,
        -0.8 + bar_width_arc/5,
        f"{bar_width_au:.0f} au",
        c="white",
        ha="center",
        fontsize=7
    )
    ax.format(
        xlim=(0.9, -0.9),
        ylim=(-0.9, 0.9),
        ylocator=[-0.6, -0.3, 0, 0.3, 0.6],
        ylabel=r'$\Delta$DEC (")',
        xlocator=[-0.6, -0.3, 0, 0.3, 0.6],
        xlabel=r'$\Delta$RA (")',
    )

    fig.savefig(paths.figures / "HD169142_ALMA_1.3mm.pdf", bbox_inches="tight")
