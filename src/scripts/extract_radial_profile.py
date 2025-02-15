import proplot as pro
import numpy as np
import paths
from astropy.io import fits
from astropy.convolution import convolve, kernels
import tqdm
from astropy.stats import biweight_location, biweight_scale
import polarTransform as pt
import pandas as pd
from astropy.nddata import Cutout2D
from utils_organization import folders, pxscales
from instrument_info import alma_info
from target_info import target_info

vampires_filters = ["F610", "F670", "F720", "F760"]
vampires_psfs = [
    fits.getdata(paths.data / f"VAMPIRES_{filt}_synthpsf.fits")
    for filt in vampires_filters
]

def crop(data, window):
    cy, cx = np.array(data.shape[-2:]) / 2 - 0.5
    cutout = Cutout2D(data, (cx, cy), window)
    return cutout.data


def get_radial_profile(image, image_err, radii):
    radii_ints = np.round(radii).astype(int)
    bins = np.arange(radii_ints.min(), radii_ints.max() + 1)
    counts = []
    errs = []
    for i in range(len(bins) - 1):
        mask = (radii >= bins[i]) & (radii <  bins[i + 1]) & np.isfinite(image)
        data = image[mask]
        mean = np.nanmean(data)
        std = np.nanstd(data)
        stderr = std / np.sqrt(data.size)
        rmserr = np.sqrt(np.nansum(image_err[mask]**2)) / data.size
        counts.append(mean)
        errs.append(np.hypot(stderr, rmserr))
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    result = {"radius": bin_centers, "profile": np.array(counts), "error": np.array(errs)}
    return result

def process_vampires(folder: str) -> None:
    date = folder.split("_")[0]
    # load data
    with fits.open(
        paths.data
        / folder
        / "optimized"
        / f"{date}_HD169142_vampires_stokes_cube_optimized.fits"
    ) as hdul:
        stokes_cube = hdul[0].data

    with fits.open(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_radius.fits"
    ) as hdul:
        radius_map = hdul[0].data / target_info.dist_pc / pxscales[folder]

    r2_map = radius_map**2


    # warp to polar coordinates
    psf = np.sum(vampires_psfs, axis=0)
    psf /= np.sum(psf)

    _data = convolve(crop(stokes_cube[4], 400), psf) * r2_map
    _err = convolve(crop(stokes_cube[5], 400), psf) * r2_map

    # print(folder, mask_name)
    # quickplot(_data, _err)
    info = get_radial_profile(_data, _err, radius_map)


    output_df = pd.DataFrame(
        {
            "radius(au)": info["radius"] * target_info.dist_pc * pxscales[folder],
            "filter": "MBI",
            "Qphi": info["profile"],
            "Qphi_err": info["error"],
        }
    )
    output_df.dropna(axis=0, how="any", inplace=True)
    output_name = paths.data / folder / f"{folder}_HD169142_radial_profiles.csv"
    output_df.to_csv(output_name, index=False)


def process_naco(folder: str) -> None:
    # load data
    Qphi = fits.getdata(
        paths.data / folder / "coadded" / "Q_phi.fits",
        ext=("Q_PHI_CTC_IPS", 1),
    )
    Uphi = fits.getdata(
        paths.data / folder / "coadded" / "U_phi.fits", ext=("U_PHI_CTC_IPS", 1)
    )

    radius_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_radius.fits"
    ) / target_info.dist_pc / pxscales[folder]
    r2_map = radius_map**2


    kernel_fwhm = 2
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    _data = convolve(crop(Qphi, 120), kernel) * r2_map
    _err = convolve(crop(Uphi, 120), kernel) * r2_map

    # print(folder, mask_name)
    # quickplot(_data, _err)
    info = get_radial_profile(_data, _err, radius_map)

    output_df = pd.DataFrame(
        {
            "radius(au)": info["radius"] * target_info.dist_pc * pxscales[folder],
            "filter": "H",
            "Qphi": info["profile"],
            "Qphi_err": info["error"],
        }
    )
    output_df.dropna(axis=0, how="any", inplace=True)
    output_name = paths.data / folder / f"{folder}_HD169142_radial_profiles.csv"
    output_df.to_csv(output_name, index=False)



def process_irdis(folder: str) -> None:
    # load data
    cube = fits.getdata(paths.data / folder / f"{folder}_HD169142_stokes_cube.fits")
    Qphi = cube[1]
    Uphi = cube[2]

    radius_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_radius.fits"
    ) / target_info.dist_pc / pxscales[folder]
    r2_map = radius_map**2

    kernel_fwhm = 1
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    _data = convolve(crop(Qphi, 500), kernel) * r2_map
    _err = convolve(crop(Uphi, 500), kernel) * r2_map
    # print(folder, mask_name)
    # quickplot(_data, _err)
    info = get_radial_profile(_data, _err, radius_map)
    _filt = "J" if "2015" in folder else "K"


    output_df = pd.DataFrame(
        {
            "radius(au)": info["radius"] * target_info.dist_pc * pxscales[folder],
            "filter": _filt,
            "Qphi": info["profile"],
            "Qphi_err": info["error"],
        }
    )
    output_df.dropna(axis=0, how="any", inplace=True)
    output_name = paths.data / folder / f"{folder}_HD169142_radial_profiles.csv"
    output_df.to_csv(output_name, index=False)


def process_zimpol(folder: str) -> None:
    # load data
    Qphi = fits.getdata(paths.data / folder / "Qphi.fits")
    Uphi = fits.getdata(paths.data / folder / "Uphi.fits")

    radius_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_radius.fits"
    ) / target_info.dist_pc / pxscales[folder]

    r2_map = radius_map**2


    kernel_fwhm = 1
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    _data = convolve(Qphi, kernel) * r2_map
    _err = convolve(Uphi, kernel) * r2_map
    # print(folder, mask_name)
    # quickplot(_data, _err)
    info = get_radial_profile(_data, _err, radius_map)

    output_df = pd.DataFrame(
        {
            "radius(au)": info["radius"] * target_info.dist_pc * pxscales[folder],
            "filter": "VBB",
            "Qphi": info["profile"],
            "Qphi_err": info["error"],
        }
    )
    output_df.dropna(axis=0, how="any", inplace=True)
    output_name = paths.data / folder / f"{folder}_HD169142_radial_profiles.csv"
    output_df.to_csv(output_name, index=False)


def process_gpi(folder: str) -> None:
    # load data
    cube = fits.getdata(paths.data / folder / f"{folder}_HD169142_stokes_cube.fits")
    Qphi = cube[1]
    Uphi = cube[2]

    radius_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_radius.fits"
    ) / target_info.dist_pc / pxscales[folder]

    r2_map = radius_map**2

    kernel_fwhm = 1
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    _data = convolve(Qphi, kernel) * r2_map
    _err = convolve(Uphi, kernel) * r2_map

    # print(folder, mask_name)
    # quickplot(_data, _err)
    info = get_radial_profile(_data, _err, radius_map)


    output_df = pd.DataFrame(
        {
            "radius(au)": info["radius"] * target_info.dist_pc * pxscales[folder],
            "filter": "J",
            "Qphi": info["profile"],
            "Qphi_err": info["error"],
        }
    )
    output_df.dropna(axis=0, how="any", inplace=True)
    output_name = paths.data / folder / f"{folder}_HD169142_radial_profiles.csv"
    output_df.to_csv(output_name, index=False)



def process_charis(folder: str) -> None:
    # load data
    Qphi = crop(fits.getdata(paths.data / folder / f"{folder}_HD169142_Qphi.fits"), 140)
    Uphi = crop(fits.getdata(paths.data / folder / f"{folder}_HD169142_Qphi.fits"), 140)

    radius_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_radius.fits"
    ) / target_info.dist_pc / pxscales[folder]

    r2_map = radius_map**2

    kernel_fwhm = 1
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    _data = convolve(Qphi, kernel) * r2_map
    _err = convolve(Uphi, kernel) * r2_map

    # print(folder, mask_name)
    # quickplot(_data, _err)
    info = get_radial_profile(_data, _err, radius_map)


    output_df = pd.DataFrame(
        {
            "radius(au)": info["radius"] * target_info.dist_pc * pxscales[folder],
            "filter": "JHK",
            "Qphi": info["profile"],
            "Qphi_err": info["error"],
        }
    )
    output_df.dropna(axis=0, how="any", inplace=True)
    output_name = paths.data / folder / f"{folder}_HD169142_radial_profiles.csv"
    output_df.to_csv(output_name, index=False)


def process_alma(folder: str) -> None:
    # load data
    frame = fits.getdata(paths.data / folder / "HD169142.selfcal.concat.GPU-UVMEM.centered_mJyBeam.fits")

    radius_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_radius.fits"
    ) / target_info.dist_pc / pxscales[folder]


    # warp to polar coordinates
    _data = frame
    _err = np.full_like(_data, alma_info.noise)
    info = get_radial_profile(_data, _err, radius_map)

    output_df = pd.DataFrame(
        {
            "radius(au)": info["radius"] * target_info.dist_pc * pxscales[folder],
            "filter": "1.3mm",
            "I": info["profile"],
            "I_err": info["error"],
        }
    )
    output_df.dropna(axis=0, how="any", inplace=True)
    output_name = paths.data / folder / f"{folder}_HD169142_radial_profiles.csv"
    output_df.to_csv(output_name, index=False)

if __name__ == "__main__":
    _folders = [*folders, "20170918_ALMA_1.3mm"]
    for i, folder in enumerate(tqdm.tqdm(_folders)):
        if "VAMPIRES" in folder:
            process_vampires(folder)
        elif "NACO" in folder:
            process_naco(folder)
        elif "IRDIS" in folder:
            process_irdis(folder)
        elif "ZIMPOL" in folder:
            process_zimpol(folder)
        elif "GPI" in folder:
            process_gpi(folder)
        elif "ALMA" in folder:
            process_alma(folder)
        elif "CHARIS" in folder:
            process_charis(folder)
        else:
            print(f"Folder not recognized: {folder=}")