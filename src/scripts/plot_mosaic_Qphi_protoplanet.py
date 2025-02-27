import paths
from astropy.io import fits
import numpy as np
import proplot as pro
from astropy.visualization import simple_norm
from utils_organization import label_from_folder, folders, pxscales
from utils_plots import setup_rc
from target_info import target_info
from utils_ephemerides import blob_d_position
from utils_organization import time_from_folder
from matplotlib import patches
from skimage import filters

def inner_ring_mask(frame, radii):
    rin_au = 15
    rout_au = 35
    rad_mask = (radii >= rin_au) & (radii <= rout_au)
    return np.where(rad_mask, frame, np.nan)

def gap_mask(frame, radii):
    rin_au = 45 - 5
    rout_au = 45 + 5
    rad_mask = (radii >= rin_au) & (radii <= rout_au)
    return np.where(rad_mask, frame, np.nan)

if __name__ == "__main__":
    setup_rc()
    pro.rc["axes.facecolor"] = "k"
    pro.rc["axes.grid"] = False

    fig, axes = pro.subplots(ncols=4, nrows=2, width="7in", hspace=1.75, wspace=0.5, spanx=False)


    for i, folder in enumerate(folders):
        stokes_path = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Qphi_deprojected.fits"
        Qphi_image, header = fits.getdata(stokes_path, header=True)
        radius_path = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Qphi_radius.fits"
        radius_map_au = fits.getdata(radius_path)

        # Qphi_image = 1.2 * filters.gaussian(Qphi_image, 1) - filters.median(Qphi_image, np.ones((10, 10)))
        # Qphi_image = filters.unsharp_mask(Qphi_image, radius=3, amount=5, preserve_range=True)

        Qphi_image = filters.gaussian(Qphi_image, 1)
        Qphi_image_masked = gap_mask(Qphi_image, radius_map_au)

        side_length = Qphi_image.shape[-1] * pxscales[folder] / 2
        ext = (side_length, -side_length, -side_length, side_length)

        vmax = np.nanmax(Qphi_image_masked)
        norm = simple_norm(Qphi_image, vmin=0, vmax=vmax, stretch="sqrt", asinh_a=0.05)
        axes[i].imshow(Qphi_image, extent=ext, cmap="bone", norm=norm, vmin=norm.vmin, vmax=norm.vmax)
        labels = label_from_folder(folder).split()
        axes[i].text(
            0.03, 1.01, labels[0],
            transform="axes",
            c="0.3",
            fontweight="bold",
            ha="left",
            va="bottom"
        )
        axes[i].text(
            0.99, 1.01, " ".join(labels[1:]),
            transform="axes",
            c="0.3",
            fontweight="bold",
            ha="right",
            va="bottom"
        )

        r, theta = blob_d_position(time_from_folder(folder))
        r_arc = r / target_info.dist_pc
        th = np.deg2rad(theta + 90)
        x = r_arc * -np.cos(th)
        y = r_arc * np.sin(th)
        rad_au = 4
        rad_arc = rad_au / target_info.dist_pc
        patch = patches.Circle((x, y), rad_arc, color="white", lw=1, fill=False)
        axes[i].add_patch(patch)
        # axes[i].scatter(x, y, marker="+", c="white", ms=40, lw=1)


    eph_r, eph_th = blob_d_position(time_from_folder("20180101"))
    eph_r /= target_info.dist_pc
    eph_x = -eph_r * np.cos(np.deg2rad(eph_th + 90))
    eph_y = eph_r * np.sin(np.deg2rad(eph_th + 90))
    window = 0.2

    axes.format(
        xlim=(eph_x + window, eph_x - window),
        ylim=(eph_y - window, eph_y + window),
        # xlocator=[0.6, 0.3, 0, -0.3, -0.6],
        # ylocator=[-0.6, -0.3, 0, 0.3, 0.6],
        # xlabel=r'$\Delta$RA (")',
        # ylabel=r'$\Delta$DEC (")',
        xlocator="none",
        ylocator="none"
    )

    # axes[1].format(yspineloc="none")

    fig.savefig(
        paths.figures / "HD169142_Qphi_mosaic_protoplanet.pdf",
        bbox_inches="tight", dpi=300
    )
    