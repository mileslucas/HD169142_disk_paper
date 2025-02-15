import numpy as np
import paths
from astropy.io import fits
from astropy.convolution import convolve, kernels
import tqdm
import pandas as pd
from astropy.nddata import Cutout2D
from target_info import target_info
from utils_ephemerides import keplerian_warp2d
from utils_organization import time_from_folder

vampires_filters = ["F610", "F670", "F720", "F760"]
vampires_psfs = [
    fits.getdata(paths.data / f"VAMPIRES_{filt}_synthpsf.fits")
    for filt in vampires_filters
]
t0 = time_from_folder("20180715_zimpol")
t0 = time_from_folder("20150710_zimpol")


def crop(data, window):
    cy, cx = np.array(data.shape[-2:]) / 2 - 0.5
    cutout = Cutout2D(data, (cx, cy), window)
    return cutout.data


def get_spf(image, image_err, scat_angle_deg, bin_width=1) -> dict:
    azimuth_ints = np.round(scat_angle_deg).astype(int)
    bins = np.arange(azimuth_ints.min(), azimuth_ints.max() + 1, bin_width)
    counts = []
    errs = []
    for i in range(len(bins) - 1):
        mask = (scat_angle_deg >= bins[i]) & (scat_angle_deg <  bins[i + 1]) & np.isfinite(image)
        data = image[mask]
        err = image_err[mask]
        mean = np.mean(data)
        counts.append(mean)
        N = data.size
        std_err = np.std(data) / np.sqrt(N)
        tot_err = np.sqrt(np.sum(err**2) / N**2 + std_err**2)
        errs.append(tot_err)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    result = {"scat_angle": bin_centers, "profile": np.array(counts), "error": np.array(errs)}
    return result

def quickplot(Qphi, Uphi):
    import proplot as pro
    from astropy.visualization import simple_norm
    fig, axes = pro.subplots(ncols=2)
    norm = simple_norm(Qphi, stretch="asinh", vmin=0)
    axes[0].imshow(Qphi, origin="lower", cmap="magma", norm=norm, vmin=0)
    axes[1].imshow(Uphi, origin="lower", cmap="magma", norm=norm, vmin=0)
    axes.format(toplabels=("Qphi", "Uphi"))
    pro.show(block=True)
    pro.close()

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
        radius_map = hdul[0].data
    with fits.open(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_scat_angle.fits"
    ) as hdul:
        scat_angle_map = hdul[0].data

    r2_map = radius_map**2

    masks = {
        "inner": (radius_map > 15) & (radius_map <= 35),
        "outer": (radius_map > 48) & (radius_map <= 110),
    }

    # I_frames = np.nanmean(stokes_cube[:, :2], axis=1)
    # Qphi_frames = stokes_cube[:, 4]
    # Uphi_frames = stokes_cube[:, 5]
    time = time_from_folder(folder)
    Qphi_profiles = []

    # warp to polar coordinates
    psf = np.sum(vampires_psfs, axis=0)
    psf /= np.sum(psf)

    for mask_name, mask in masks.items():
        _data = convolve(crop(stokes_cube[4], 400), psf) * r2_map
        _err = convolve(crop(stokes_cube[5], 400), psf) * r2_map
        _data = keplerian_warp2d(_data, radius_map, time, t0)
        _err = keplerian_warp2d(_err, radius_map, time, t0)
        _data[~mask] = np.nan
        _err[~mask] = np.nan
        # print(folder, mask_name)
        # quickplot(_data, _err)
        info = get_spf(_data, _err, scat_angle_map)
        info["filter"] = "MBI"
        info["region"] = mask_name
        Qphi_profiles.append(info)

    Qphi_dataframe = pd.concat(map(pd.DataFrame, Qphi_profiles))

    output_df = pd.DataFrame(
        {
            "scat_angle(deg)": Qphi_dataframe["scat_angle"],
            "filter": Qphi_dataframe["filter"],
            "region": Qphi_dataframe["region"],
            "Qphi": Qphi_dataframe["profile"],
            "Qphi_err": Qphi_dataframe["error"],
        }
    )
    output_df.dropna(axis=0, how="any", inplace=True)
    output_name = paths.data / folder / f"{folder}_HD169142_pol_spf.csv"
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
    scat_angle_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_scat_angle.fits"
    )
    r2_map = radius_map**2

    masks = {
        "inner": (radius_map > 15) & (radius_map <= 35),
        "outer": (radius_map > 48) & (radius_map <= 110),
    }

    time = time_from_folder(folder)
    Qphi_profiles = []
    kernel_fwhm = 2
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    for mask_name, mask in masks.items():
        _data = convolve(crop(Qphi, 120), kernel) * r2_map
        _err = convolve(crop(Uphi, 120), kernel) * r2_map
        _data = keplerian_warp2d(_data, radius_map, time, t0)
        _err = keplerian_warp2d(_err, radius_map, time, t0)
        _data[~mask] = np.nan
        _err[~mask] = np.nan
        # print(folder, mask_name)
        # quickplot(_data, _err)
        info = get_spf(_data, _err, scat_angle_map)
        info["filter"] = "H"
        info["region"] = mask_name
        Qphi_profiles.append(info)

    Qphi_dataframe = pd.concat(map(pd.DataFrame, Qphi_profiles))

    output_df = pd.DataFrame(
        {
            "scat_angle(deg)": Qphi_dataframe["scat_angle"],
            "filter": Qphi_dataframe["filter"],
            "region": Qphi_dataframe["region"],
            "Qphi": Qphi_dataframe["profile"],
            "Qphi_err": Qphi_dataframe["error"],
        }
    )
    output_df.dropna(axis=0, how="any", inplace=True)
    output_name = paths.data / folder / f"{folder}_HD169142_pol_spf.csv"
    output_df.to_csv(output_name, index=False)


def process_irdis(folder: str) -> None:
    # load data
    cube = fits.getdata(paths.data / folder / f"{folder}_HD169142_stokes_cube.fits")
    Qphi = cube[1]
    Uphi = cube[2]

    radius_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_radius.fits"
    )
    scat_angle_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_scat_angle.fits"
    )
    r2_map = radius_map**2

    masks = {
        "inner": (radius_map > 15) & (radius_map <= 35),
        "outer": (radius_map > 48) & (radius_map <= 110),
    }

    time = time_from_folder(folder)
    Qphi_profiles = []
    kernel_fwhm = 1
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    for mask_name, mask in masks.items():
        _data = convolve(crop(Qphi, 500), kernel) * r2_map
        _err = convolve(crop(Uphi, 500), kernel) * r2_map
        _data = keplerian_warp2d(_data, radius_map, time, t0)
        _err = keplerian_warp2d(_err, radius_map, time, t0)
        _data[~mask] = np.nan
        _err[~mask] = np.nan
        # print(folder, mask_name)
        # quickplot(_data, _err)
        info = get_spf(_data, _err, scat_angle_map)
        info["filter"] = "J" if "2015" in folder else "K"
        info["region"] = mask_name
        Qphi_profiles.append(info)

    Qphi_dataframe = pd.concat(map(pd.DataFrame, Qphi_profiles))

    output_df = pd.DataFrame(
        {
            "scat_angle(deg)": Qphi_dataframe["scat_angle"],
            "filter": Qphi_dataframe["filter"],
            "region": Qphi_dataframe["region"],
            "Qphi": Qphi_dataframe["profile"],
            "Qphi_err": Qphi_dataframe["error"],
        }
    )
    output_df.dropna(axis=0, how="any", inplace=True)
    output_name = paths.data / folder / f"{folder}_HD169142_pol_spf.csv"
    output_df.to_csv(output_name, index=False)


def process_zimpol(folder: str) -> None:
    # load data
    Qphi = fits.getdata(paths.data / folder / "Qphi.fits")
    Uphi = fits.getdata(paths.data / folder / "Uphi.fits")

    radius_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_radius.fits"
    )
    scat_angle_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_scat_angle.fits"
    )
    r2_map = radius_map**2

    masks = {
        "inner": (radius_map > 15) & (radius_map <= 35),
        "outer": (radius_map > 48) & (radius_map <= 110),
    }

    time = time_from_folder(folder)
    Qphi_profiles = []
    kernel_fwhm = 1
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    for mask_name, mask in masks.items():
        _data = convolve(Qphi, kernel) * r2_map
        _err = convolve(Uphi, kernel) * r2_map
        _data = keplerian_warp2d(_data, radius_map, time, t0)
        _err = keplerian_warp2d(_err, radius_map, time, t0)
        _data[~mask] = np.nan
        _err[~mask] = np.nan
        # print(folder, mask_name)
        # quickplot(_data, _err)
        info = get_spf(_data, _err, scat_angle_map)
        info["filter"] = "VBB"
        info["region"] = mask_name
        Qphi_profiles.append(info)

    Qphi_dataframe = pd.concat(map(pd.DataFrame, Qphi_profiles))

    output_df = pd.DataFrame(
        {
            "scat_angle(deg)": Qphi_dataframe["scat_angle"],
            "filter": Qphi_dataframe["filter"],
            "region": Qphi_dataframe["region"],
            "Qphi": Qphi_dataframe["profile"],
            "Qphi_err": Qphi_dataframe["error"],
        }
    )
    output_df.dropna(axis=0, how="any", inplace=True)
    output_name = paths.data / folder / f"{folder}_HD169142_pol_spf.csv"
    output_df.to_csv(output_name, index=False)


def process_gpi(folder: str) -> None:
    # load data
    cube = fits.getdata(paths.data / folder / f"{folder}_HD169142_stokes_cube.fits")
    Qphi = cube[1]
    Uphi = cube[2]

    radius_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_radius.fits"
    )
    scat_angle_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_scat_angle.fits"
    )
    r2_map = radius_map**2

    masks = {
        "inner": (radius_map > 15) & (radius_map <= 35),
        "outer": (radius_map > 48) & (radius_map <= 110),
    }

    time = time_from_folder(folder)
    Qphi_profiles = []
    kernel_fwhm = 1
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    for mask_name, mask in masks.items():
        _data = convolve(Qphi, kernel) * r2_map
        _err = convolve(Uphi, kernel) * r2_map
        _data = keplerian_warp2d(_data, radius_map, time, t0)
        _err = keplerian_warp2d(_err, radius_map, time, t0)
        _data[~mask] = np.nan
        _err[~mask] = np.nan
        # print(folder, mask_name)
        # quickplot(_data, _err)
        info = get_spf(_data, _err, scat_angle_map)
        info["filter"] = "J"
        info["region"] = mask_name
        Qphi_profiles.append(info)

    Qphi_dataframe = pd.concat(map(pd.DataFrame, Qphi_profiles))

    output_df = pd.DataFrame(
        {
            "scat_angle(deg)": Qphi_dataframe["scat_angle"],
            "filter": Qphi_dataframe["filter"],
            "region": Qphi_dataframe["region"],
            "Qphi": Qphi_dataframe["profile"],
            "Qphi_err": Qphi_dataframe["error"],
        }
    )
    output_df.dropna(axis=0, how="any", inplace=True)
    output_name = paths.data / folder / f"{folder}_HD169142_pol_spf.csv"
    output_df.to_csv(output_name, index=False)

def process_charis(folder: str) -> None:
    # load data
    Qphi = fits.getdata(paths.data / folder / f"{folder}_HD169142_Qphi.fits")
    Uphi = fits.getdata(paths.data / folder / f"{folder}_HD169142_Uphi.fits")

    Qphi = crop(Qphi, 140)
    Uphi = crop(Uphi, 140)

    radius_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_radius.fits"
    )
    scat_angle_map = fits.getdata(
        paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_scat_angle.fits"
    )
    r2_map = radius_map**2

    masks = {
        "inner": (radius_map > 15) & (radius_map <= 35),
        "outer": (radius_map > 48) & (radius_map <= 110),
    }

    time = time_from_folder(folder)
    Qphi_profiles = []
    kernel_fwhm = 1
    kernel = kernels.Gaussian2DKernel(kernel_fwhm / (2 * np.sqrt(2 * np.log(2))))
    # warp to polar coordinates
    for mask_name, mask in masks.items():
        _data = convolve(Qphi, kernel) * r2_map
        _err = convolve(Uphi, kernel) * r2_map
        _data = keplerian_warp2d(_data, radius_map, time, t0)
        _err = keplerian_warp2d(_err, radius_map, time, t0)
        _data[~mask] = np.nan
        _err[~mask] = np.nan
        # print(folder, mask_name)
        # quickplot(_data, _err)
        info = get_spf(_data, _err, scat_angle_map)
        info["filter"] = "JHK"
        info["region"] = mask_name
        Qphi_profiles.append(info)

    Qphi_dataframe = pd.concat(map(pd.DataFrame, Qphi_profiles))

    output_df = pd.DataFrame(
        {
            "scat_angle(deg)": Qphi_dataframe["scat_angle"],
            "filter": Qphi_dataframe["filter"],
            "region": Qphi_dataframe["region"],
            "Qphi": Qphi_dataframe["profile"],
            "Qphi_err": Qphi_dataframe["error"],
        }
    )
    output_df.dropna(axis=0, how="any", inplace=True)
    output_name = paths.data / folder / f"{folder}_HD169142_pol_spf.csv"
    output_df.to_csv(output_name, index=False)



if __name__ == "__main__":
    folders = [
        "20120726_NACO_H",
        "20140425_GPI_J",
        "20150503_IRDIS_J",
        "20150710_ZIMPOL_VBB",
        "20180715_ZIMPOL_VBB",
        "20210906_IRDIS_Ks",
        "20230604_CHARIS_JHK",
        "20230707_VAMPIRES_MBI",
        "20240729_VAMPIRES_MBI",
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
        elif "CHARIS" in folder:
            process_charis(folder)
        else:
            print(f"Folder not recognized: {folder=}")