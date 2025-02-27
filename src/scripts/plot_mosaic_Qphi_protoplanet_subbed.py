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
from vampires_dpp.image_processing import derotate_frame
import tqdm

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


    for i, folder in enumerate(tqdm.tqdm(folders, desc="Simulating ADI images for cADI")):
        Qphi_image_subbed = fits.getdata(paths.data / folder / f"{folder}_HD169142_Qphi_cADI_sim.fits")
        radius_path = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Qphi_radius.fits"
        radius_map_au = fits.getdata(radius_path)

        Qphi_image_subbed = filters.gaussian(Qphi_image_subbed, 1)

        Qphi_image_masked = gap_mask(Qphi_image_subbed, radius_map_au)

        side_length = Qphi_image_subbed.shape[-1] * pxscales[folder] / 2
        ext = (side_length, -side_length, -side_length, side_length)

        # vmax = np.nanpercentile(Qphi_image_masked, 99.9)
        vmax = np.nanmax(Qphi_image_masked)
        # norm = simple_norm(Qphi_image_subbed, vmin=0, vmax=vmax, stretch="sqrt", asinh_a=0.05)
        norm = pro.DivergingNorm(vmin=-vmax, vmax=vmax)
        axes[i].imshow(Qphi_image_subbed, extent=ext, cmap="div", norm=norm, vmin=norm.vmin, vmax=norm.vmax)
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
        patch = patches.Circle((x, y), 0.05, color="white", lw=1, fill=False)
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
        paths.figures / "HD169142_Qphi_mosaic_protoplanet_subbed.pdf",
        bbox_inches="tight", dpi=300
    )
    