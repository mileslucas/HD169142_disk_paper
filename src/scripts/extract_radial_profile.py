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
    for bin in bins:
        mask = (radii_ints == bin) & np.isfinite(image)
        data = image[mask]
        mean = biweight_location(data)
        counts.append(mean)
        errs.append(biweight_scale(data))
    result = {"radius": bins, "profile": np.array(counts), "error": np.array(errs)}
    return result

def process_vampires(folder: str) -> None:
    date = folder.replace("_VAMPIRES", "")
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
        radius_map = hdul[0].data

    r2_map = radius_map**2

    Qphi_profiles = []

    # warp to polar coordinates
    psf = np.sum(vampires_psfs, axis=0)
    psf /= np.sum(psf)

    _data = convolve(crop(stokes_cube[4], 400), psf) * r2_map
    _err = convolve(crop(stokes_cube[5], 400), psf) * r2_map

    # print(folder, mask_name)
    # quickplot(_data, _err)
    info = get_radial_profile(_data, _err, radius_map)
    info["filter"] = "MBI"
    Qphi_profiles.append(info)

    Qphi_dataframe = pd.concat(map(pd.DataFrame, Qphi_profiles))

    output_df = pd.DataFrame(
        {
            "radius(au)": Qphi_dataframe["radius"],
            "filter": Qphi_dataframe["filter"],
            "Qphi": Qphi_dataframe["profile"],
            "Qphi_err": Qphi_dataframe["error"],
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
    )
    r2_map = radius_map**2


    Qphi_profiles = []
    kernel_fwhm = 2
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    _data = convolve(crop(Qphi, 120), kernel) * r2_map
    _err = convolve(crop(Uphi, 120), kernel) * r2_map

    # print(folder, mask_name)
    # quickplot(_data, _err)
    info = get_radial_profile(_data, _err, radius_map)
    info["filter"] = "H"
    Qphi_profiles.append(info)

    Qphi_dataframe = pd.concat(map(pd.DataFrame, Qphi_profiles))

    output_df = pd.DataFrame(
        {
            "radius(au)": Qphi_dataframe["radius"],
            "filter": Qphi_dataframe["filter"],
            "Qphi": Qphi_dataframe["profile"],
            "Qphi_err": Qphi_dataframe["error"],
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
    )
    r2_map = radius_map**2

    Qphi_profiles = []
    kernel_fwhm = 1
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    _data = convolve(crop(Qphi, 500), kernel) * r2_map
    _err = convolve(crop(Uphi, 500), kernel) * r2_map
    # print(folder, mask_name)
    # quickplot(_data, _err)
    info = get_radial_profile(_data, _err, radius_map)
    info["filter"] = "J" if "2015" in folder else "K"
    Qphi_profiles.append(info)

    Qphi_dataframe = pd.concat(map(pd.DataFrame, Qphi_profiles))

    output_df = pd.DataFrame(
        {
            "radius(au)": Qphi_dataframe["radius"],
            "filter": Qphi_dataframe["filter"],
            "Qphi": Qphi_dataframe["profile"],
            "Qphi_err": Qphi_dataframe["error"],
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
    )

    r2_map = radius_map**2


    Qphi_profiles = []
    kernel_fwhm = 1
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    _data = convolve(Qphi, kernel) * r2_map
    _err = convolve(Uphi, kernel) * r2_map
    # print(folder, mask_name)
    # quickplot(_data, _err)
    info = get_radial_profile(_data, _err, radius_map)
    info["filter"] = "VBB"
    Qphi_profiles.append(info)

    Qphi_dataframe = pd.concat(map(pd.DataFrame, Qphi_profiles))

    output_df = pd.DataFrame(
        {
            "radius(au)": Qphi_dataframe["radius"],
            "filter": Qphi_dataframe["filter"],
            "Qphi": Qphi_dataframe["profile"],
            "Qphi_err": Qphi_dataframe["error"],
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
    )

    r2_map = radius_map**2

    Qphi_profiles = []
    kernel_fwhm = 1
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    _data = convolve(Qphi, kernel) * r2_map
    _err = convolve(Uphi, kernel) * r2_map

    # print(folder, mask_name)
    # quickplot(_data, _err)
    info = get_radial_profile(_data, _err, radius_map)
    info["filter"] = "J"
    Qphi_profiles.append(info)

    Qphi_dataframe = pd.concat(map(pd.DataFrame, Qphi_profiles))

    output_df = pd.DataFrame(
        {
            "radius(au)": Qphi_dataframe["radius"],
            "filter": Qphi_dataframe["filter"],
            "Qphi": Qphi_dataframe["profile"],
            "Qphi_err": Qphi_dataframe["error"],
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
    )

    r2_map = radius_map**2

    Qphi_profiles = []
    kernel_fwhm = 1
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    _data = convolve(Qphi, kernel) * r2_map
    _err = convolve(Uphi, kernel) * r2_map

    # print(folder, mask_name)
    # quickplot(_data, _err)
    info = get_radial_profile(_data, _err, radius_map)
    info["filter"] = "JHK"
    Qphi_profiles.append(info)

    Qphi_dataframe = pd.concat(map(pd.DataFrame, Qphi_profiles))

    output_df = pd.DataFrame(
        {
            "radius(au)": Qphi_dataframe["radius"],
            "filter": Qphi_dataframe["filter"],
            "Qphi": Qphi_dataframe["profile"],
            "Qphi_err": Qphi_dataframe["error"],
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
    )


    Qphi_profiles = []
    # warp to polar coordinates
    _data = frame
    _err = np.sqrt(frame)
    info = get_radial_profile(_data, _err, radius_map)
    info["filter"] = "1.3mm"
    Qphi_profiles.append(info)

    Qphi_dataframe = pd.concat(map(pd.DataFrame, Qphi_profiles))

    output_df = pd.DataFrame(
        {
            "radius(au)": Qphi_dataframe["radius"],
            "filter": Qphi_dataframe["filter"],
            "I": Qphi_dataframe["profile"],
            "I_err": Qphi_dataframe["error"],
        }
    )
    output_df.dropna(axis=0, how="any", inplace=True)
    output_name = paths.data / folder / f"{folder}_HD169142_radial_profiles.csv"
    output_df.to_csv(output_name, index=False)

if __name__ == "__main__":
    folders = [
        # "20120726_NACO",
        # "20140425_GPI",
        # "20150503_IRDIS",
        # "20150710_ZIMPOL",
        # "20170918_ALMA",
        # "20180715_ZIMPOL",
        # "20210906_IRDIS",
        "20230604_CHARIS",
        # "20230707_VAMPIRES",
        # "20240729_VAMPIRES",
    ]
    for i, folder in enumerate(tqdm.tqdm(folders)):
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