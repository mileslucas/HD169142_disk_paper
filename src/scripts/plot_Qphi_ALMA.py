import paths
from astropy.io import fits
import numpy as np
import proplot as pro
from astropy.visualization import simple_norm
from utils_organization import label_from_folder, folders, pxscales
from utils_plots import setup_rc

def inner_ring_mask(frame, radii):
    rin_au = 15
    rout_au = 35
    rad_mask = (radii >= rin_au) & (radii <= rout_au)
    return np.where(rad_mask, frame, np.nan)


if __name__ == "__main__":
    setup_rc()
    pro.rc["axes.facecolor"] = "k"
    pro.rc["axes.grid"] = False

    alma_data, alma_hdr = fits.getdata(paths.data / "20170918_ALMA_1.3mm" / "HD169142.selfcal.concat.GPU-UVMEM.centered_mJyBeam.fits", header=True)
    alma_pxscale = np.abs(alma_hdr["CDELT1"]) * 3.6e3 # arcsec / px
    alma_side_length = alma_data.shape[-1] * alma_pxscale / 2
    alma_ext = (alma_side_length, -alma_side_length, -alma_side_length, alma_side_length)
    alma_ys = np.linspace(alma_ext[0], alma_ext[1], alma_data.shape[0])
    alma_xs = np.linspace(alma_ext[2], alma_ext[3], alma_data.shape[1])

    fig, axes = pro.subplots(ncols=4, nrows=2, width="7in", hspace=1.75, wspace=0.5, spanx=False)


    for i, folder in enumerate(folders):
        stokes_path = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_r2_scaled.fits"
        Qphi_image, header = fits.getdata(stokes_path, header=True)
        radius_path = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_radius.fits"
        radius_map_au = fits.getdata(radius_path)

        Qphi_image_masked = inner_ring_mask(Qphi_image, radius_map_au)

        side_length = Qphi_image.shape[-1] * pxscales[folder] / 2
        ext = (side_length, -side_length, -side_length, side_length)

        vmax = np.nanmax(Qphi_image_masked)
        norm = simple_norm(Qphi_image, vmin=0, vmax=vmax, stretch="asinh")
        axes[i].imshow(Qphi_image, extent=ext, origin="lower", cmap="bone", norm=norm, vmin=norm.vmin, vmax=norm.vmax)
        labels = label_from_folder(folder).split()
        axes[i].text(
            0.03, 1.01, labels[0],
            transform="axes",
            c="0.3",
            fontsize=pro.rc["title.size"],
            fontweight="bold",
            ha="left",
            va="bottom"
        )
        axes[i].text(
            0.99, 1.01, " ".join(labels[1:]),
            transform="axes",
            c="0.3",
            fontsize=pro.rc["title.size"],
            fontweight="bold",
            ha="right",
            va="bottom"
        )


    axes.format(
        xlim=(0.9, -0.9),
        ylim=(-0.9, 0.9),
        # xlocator=[0.6, 0.3, 0, -0.3, -0.6],
        # ylocator=[-0.6, -0.3, 0, 0.3, 0.6],
        # xlabel=r'$\Delta$RA (")',
        # ylabel=r'$\Delta$DEC (")',
        xlocator="none",
        ylocator="none"
    )

    # axes[1].format(yspineloc="none")

    fig.savefig(
        paths.figures / "HD169142_Qphi_mosaic.pdf",
        bbox_inches="tight", dpi=300
    )
    levels = np.geomspace(0.05, np.nanmax(alma_data), 5)
    for ax in axes:
        ax.contour(alma_xs, alma_ys, alma_data, origin="lower", colors="white", alpha=0.5, levels=levels, lw=0.5)

    fig.savefig(
        paths.figures / "HD169142_Qphi_ALMA_mosaic.pdf",
        bbox_inches="tight", dpi=300
    )