import paths
from astropy.io import fits
from utils_plot_mosaic import plot_mosaic, plot_rdi_mosaic
import numpy as np
import proplot as pro
from astropy.visualization import simple_norm
from matplotlib import ticker
from target_info import target_info

def inner_ring_mask(frame, radii):
    rin_au = 15
    rout_au = 35
    rad_mask = (radii >= rin_au) & (radii <= rout_au)
    return np.where(rad_mask, frame, np.nan)

if __name__ == "__main__":
    iwas = {
        "20230707": 105,
        "20240727": 59,
        "20240728": 59,
        "20240729": 59
    }
    pxscale = 5.9e-3
    dist = 114.8


    alma_data, alma_hdr = fits.getdata(paths.data / "HD169142.selfcal.concat.GPU-UVMEM.centered_mJyBeam.fits", header=True)
    alma_pxscale = np.abs(alma_hdr["CDELT1"]) * 3.6e3 # arcsec / px
    alma_side_length = alma_data.shape[-1] * alma_pxscale / 2
    alma_ext = (alma_side_length, -alma_side_length, -alma_side_length, alma_side_length)
    alma_ys = np.linspace(alma_ext[0], alma_ext[1], alma_data.shape[0])
    alma_xs = np.linspace(alma_ext[2], alma_ext[3], alma_data.shape[1])

    folders = [
        "20120726_NACO",
        "20140425_GPI",
        "20150503_IRDIS",
        "20150710_ZIMPOL",
        "20180715_ZIMPOL",
        "20210906_IRDIS",
        "20230707_VAMPIRES",
        "20240729_VAMPIRES",
    ]
    def label_from_folder(foldername):
        tokens = foldername.split("_")
        date = f"{tokens[0][:4]}/{tokens[0][4:6]}/{tokens[0][6:]}"
        return f"{date} {tokens[1]}"

    labels = [label_from_folder(f) for f in folders]

    pxscales = {
        "20120726_NACO": 27e-3,
        "20140425_GPI": 14.14e-3,
        "20150503_IRDIS": 12.25e-3,
        "20150710_ZIMPOL": 3.6e-3,
        "20180715_ZIMPOL": 3.6e-3,
        "20230707_VAMPIRES": 5.9e-3,
        "20210906_IRDIS": 12.25e-3,
        "20240729_VAMPIRES": 5.9e-3,
    }

    fig, axes = pro.subplots(ncols=4, nrows=2, width="7in", wspace=0.5, spanx=False)


    for i, folder in enumerate(folders):
        stokes_path = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_r2_scaled.fits"
        Qphi_image, header = fits.getdata(stokes_path, header=True)
        radius_path = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_radius.fits"
        radius_map_au = fits.getdata(radius_path)

        Qphi_image_masked = inner_ring_mask(Qphi_image, radius_map_au)

        side_length = Qphi_image.shape[-1] * pxscales[folder] / 2
        ext = (side_length, -side_length, -side_length, side_length)

        vmax = np.nanmax(Qphi_image_masked)
        norm = simple_norm(Qphi_image, vmin=0, vmax=vmax, stretch="asinh", asinh_a=0.5)
        axes[i].imshow(Qphi_image, extent=ext, origin="lower", cmap="bone", norm=norm, vmin=norm.vmin, vmax=norm.vmax)
        axes[i].format(title=labels[i])


    axes.format(
        xlim=(0.32, -0.32),
        ylim=(-0.32, 0.32),
        # xlocator=[0.6, 0.3, 0, -0.3, -0.6],
        # ylocator=[-0.6, -0.3, 0, 0.3, 0.6],
        # xlabel=r'$\Delta$RA (")',
        # ylabel=r'$\Delta$DEC (")',
        toplabelsize=7,
        xlocator="none",
        ylocator="none"
    )

    # axes[1].format(yspineloc="none")

    fig.savefig(
        paths.figures / "HD169142_Qphi_mosaic_inner.pdf",
        bbox_inches="tight", dpi=300
    )
    levels = np.geomspace(0.05, np.nanmax(alma_data), 5)
    for ax in axes:
        ax.contour(alma_xs, alma_ys, alma_data, origin="lower", colors="white", alpha=0.5, levels=levels, lw=0.5)

    fig.savefig(
        paths.figures / "HD169142_Qphi_ALMA_mosaic_inner.pdf",
        bbox_inches="tight", dpi=300
    )