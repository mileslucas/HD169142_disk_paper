import proplot as pro
import numpy as np
import paths
from astropy.io import fits
from skimage.transform import warp_polar
from astropy.convolution import convolve, kernels
import tqdm
from astropy.visualization import simple_norm
from target_info import target_info
from astropy import time
from scipy import interpolate

from utils_plots import setup_rc
from utils_ephemerides import keplerian_warp
from utils_organization import folders, pxscales, time_from_folder

if __name__ == "__main__": 
    setup_rc()

    common_rs = np.linspace(15, 35, 100)
    common_thetas = np.arange(0, 361)
    thetas_grid, rs_grid = np.meshgrid(common_thetas, common_rs)

    alma_folder = "20170918_ALMA_1.3mm"
    alma_data = fits.getdata(paths.data / alma_folder / f"{alma_folder}_HD169142_Qphi_polar.fits")
    rs = np.arange(alma_data.shape[0])
    rin = np.floor(15 / target_info.dist_pc / pxscales[alma_folder]).astype(int)
    rout = np.ceil(35 / target_info.dist_pc / pxscales[alma_folder]).astype(int)
    mask = (rs >= rin) & (rs <= rout)
    alma_data = alma_data[mask, :]
    rs_au = rs[mask] * target_info.dist_pc * pxscales[alma_folder]


    this_thetas = np.linspace(0, 360, alma_data.shape[1])
    _ths, _rs = np.meshgrid(this_thetas, rs_au)
    alma_data = interpolate.griddata((_rs.ravel(), _ths.ravel()), alma_data.ravel(), (rs_grid.ravel(), thetas_grid.ravel()), method="cubic").reshape((len(common_rs), len(common_thetas)))

    alma_curve = np.nanmean(alma_data, axis=0)
    alma_curve = alma_curve / alma_curve.mean() - 1
    alma_err = np.nanstd(alma_data, axis=0)

    alma_time = time_from_folder(alma_folder)
    
    ## Plot and save
    width = 3.31314
    aspect_ratio = 1/1.6
    height = width * aspect_ratio
    fig, axes = pro.subplots(
        width=f"{width}in", height=f"{height}in"
    )

    images= []
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
        rs_au = rs[mask] * target_info.dist_pc * pxscales[folder]

        this_time = time_from_folder(folder)
        polar_cube_warped = keplerian_warp(polar_cube[mask, :], rs_au, this_time, alma_time)

        this_thetas = np.linspace(0, 360, polar_cube_warped.shape[1])
        _ths, _rs = np.meshgrid(this_thetas, rs_au)
        data = interpolate.griddata((_rs.ravel(), _ths.ravel()), polar_cube_warped.ravel(), (rs_grid.ravel(), thetas_grid.ravel()), method="cubic").reshape((len(common_rs), len(common_thetas)))
        images.append(data / np.nanmedian(data))


    data = np.nanmean(images, axis=0)
    # PDI images
    levels = np.nanpercentile(data, [80, 90, 96.5])
    im = axes[0].contour(thetas_grid, rs_grid, data, c="C0", levels=levels, zorder=10)

    levels = np.nanpercentile(alma_data, [60, 75, 90])
    im = axes[0].contour(thetas_grid, rs_grid, alma_data, c="C3", levels=levels, zorder=5)

    axes[0].text(
        0.98, 0.02,
        r"Mean $Q_\phi \times r^2$",
        c="C0",
        fontweight="bold",
        transform="axes",
        ha="right",
        va="bottom",
    )
    axes[0].text(
        0.98, 0.98,
        "ALMA (1.3mm)",
        c="C3",
        fontweight="bold",
        transform="axes",
        ha="right",
        va="top",
    )

    ## sup title
    axes.format(
        aspect="auto",
        xlabel="Angle E of N (°)",
        ylabel="Separation (au)",
        xlocator=90,
    ) 


    fig.savefig(
        paths.figures / "HD169142_polar_median_ALMA_contours.pdf", bbox_inches="tight", dpi=300
    )


