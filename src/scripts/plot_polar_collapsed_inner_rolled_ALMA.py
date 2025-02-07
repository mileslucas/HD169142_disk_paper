import proplot as pro
import numpy as np
import paths
from astropy.io import fits
from skimage.transform import warp_polar
from astropy.convolution import convolve, kernels
import tqdm
from astropy.visualization import simple_norm
from target_info import target_info
from scipy import interpolate

from utils_organization import folders, pxscales, time_from_folder
from utils_ephemerides import keplerian_warp
from utils_plots import setup_rc

if __name__ == "__main__": 
    setup_rc()
    pro.rc["axes.grid"] = False
    pro.rc["axes.facecolor"] = "k"

    alma_folder = "20170918_ALMA_1.3mm"
    alma_data = fits.getdata(paths.data / alma_folder / f"{alma_folder}_HD169142_Qphi_polar.fits")
    rs = np.arange(alma_data.shape[0])
    rin = np.floor(15 / target_info.dist_pc / pxscales[alma_folder]).astype(int)
    rout = np.ceil(35 / target_info.dist_pc / pxscales[alma_folder]).astype(int)
    mask = (rs >= rin) & (rs <= rout)
    alma_data = alma_data[mask, :]
    alma_curve = np.nanmean(alma_data, axis=0)
    alma_curve = alma_curve / alma_curve.mean() - 1
    alma_err = np.nanstd(alma_data, axis=0)

    alma_time = time_from_folder(alma_folder)
    
    ## Plot and save
    width = 3.31314
    aspect_ratio = 1/1.6
    height = width * aspect_ratio
    fig, axes = pro.subplots(
        nrows=2, width=f"{width}in", height=f"{height}in", hspace=0.25
    )

    common_rs = np.linspace(15, 35, 100)
    common_thetas = np.arange(0, 361)
    thetas_grid, rs_grid = np.meshgrid(common_thetas, common_rs)
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
        polar_cube_rolled = keplerian_warp(polar_cube[mask, :], rs_au, this_time, alma_time)

        this_thetas = np.linspace(0, 360, polar_cube_rolled.shape[1])
        _ths, _rs = np.meshgrid(this_thetas, rs_au)
        data = interpolate.griddata((_rs.ravel(), _ths.ravel()), polar_cube_rolled.ravel(), (rs_grid.ravel(), thetas_grid.ravel()), method="cubic").reshape((len(common_rs), len(common_thetas)))
        images.append(data / np.nanmedian(data))


    norm = simple_norm(alma_data, vmin=0)#, stretch="sinh", sinh_a=0.5)
    im = axes[0].imshow(alma_data, extent=ext, norm=norm, vmin=norm.vmin, vmax=norm.vmax, cmap="magma")

    data = np.nanmean(images, axis=0)
    # PDI images
    norm = simple_norm(data, vmin=0, stretch="sinh", sinh_a=0.5)
    im = axes[1].imshow(data, extent=ext, norm=norm, vmin=norm.vmin, vmax=norm.vmax, cmap=pro.rc["cmap"])

    axes[0].text(
        0.01, 0.95, "ALMA (1.3 mm)", c="white", ha="left", va="top", transform="axes"
    )
    axes[1].text(
        0.01, 0.95, r"Mean $Q_\phi \times r^2$", c="white", ha="left", va="top", transform="axes"
    )

    ## sup title
    axes.format(
        aspect="auto",
        xlabel="Angle E of N (°)",
        ylabel="Separation (au)",
        xlocator=90,
    ) 

    axes[:-1].format(xtickloc="none")

    fig.savefig(
        paths.figures / "HD169142_polar_median_ALMA.pdf", bbox_inches="tight"
    )


