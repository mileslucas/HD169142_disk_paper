import paths
from astropy.io import fits
import numpy as np
import proplot as pro
from astropy.visualization import simple_norm
from utils_organization import label_from_folder, folders, pxscales, wavelengths
from utils_plots import setup_rc
from target_info import target_info
from utils_ephemerides import blob_d_position
from utils_organization import time_from_folder
from utils_indexing import frame_radii
from matplotlib import patches
from skimage import filters
from fake_adi_subtraction import create_radial_noise_image
from photutils.aperture import CircularAperture, ApertureStats

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

    width = 3.31314
    aspect_ratio = 1/1.66
    height = width * aspect_ratio
    fig, axes = pro.subplots(width=f"{width}in", height=f"{height}in")

    for i, folder in enumerate(folders):
        Qphi_path = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Qphi_r2_scaled.fits"
        Qphi_image, header = fits.getdata(Qphi_path, header=True)
        Uphi_path = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Uphi_r2_scaled.fits"
        Uphi_image, header = fits.getdata(Uphi_path, header=True)
        radius_map = frame_radii(Qphi_image)
        radius_map_au = radius_map * pxscales[folder] * target_info.dist_pc

        sep_au, theta = blob_d_position(time_from_folder(folder))
        sep_px = sep_au / target_info.dist_pc / pxscales[folder]
        th = np.deg2rad(theta + 90)
        x = sep_px * np.cos(th)
        y = sep_px * np.sin(th)
        rad_au = 2
        rad_px = rad_au / target_info.dist_pc / pxscales[folder]
        cy, cx = np.array(Qphi_image.shape) / 2 - 0.5
        aperture = CircularAperture((x + cx, y + cy), r=rad_px)

        phot_Qphi = ApertureStats(Qphi_image, aperture, error=Uphi_image)

        snr = phot_Qphi.sum / phot_Qphi.sum_err

        wl = wavelengths[folder]
        axes[0].scatter(wl, snr)


    # eph_r, eph_th = blob_d_position(time_from_folder("20180101"))
    # eph_r /= target_info.dist_pc
    # eph_x = -eph_r * np.cos(np.deg2rad(eph_th + 90))
    # eph_y = eph_r * np.sin(np.deg2rad(eph_th + 90))
    # window = 0.2

    axes.format(
        # ylim=(0, 10),
        xlabel=r"$\lambda$ ($\mu$m)",
        ylabel=r"$Q_\phi$ S/N"
    )

    # axes[1].format(yspineloc="none")

    # fig.savefig(
    #     paths.figures / "HD169142_Qphi_mosaic_protoplanet.pdf",
    #     bbox_inches="tight", dpi=300
    # )
    
    pro.show(block=True)