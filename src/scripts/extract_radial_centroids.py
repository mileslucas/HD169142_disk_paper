import numpy as np
import pandas as pd
import paths
import tqdm
from astropy.io import fits
from target_info import target_info
from utils_errorprop import bootstrap_argmax_and_max
from utils_organization import folders, pxscales


def centroid_with_errors(x, y, err, mask=None):
    _mask = np.isfinite(y) & np.isfinite(err)
    if mask is not None:
        _mask &= mask

    _x = x[_mask]
    _y = y[_mask]
    _err = err[_mask]

    N = np.sum(_x * _y)
    D = np.sum(_y)
    x_com = N / D

    d_x_com_df = (x * D - N) / D**2
    sigma_x_com = np.sqrt(np.sum((d_x_com_df * _err) ** 2))

    return x_com, sigma_x_com


def get_centroids(folder: str, rmin: float, rmax: float) -> pd.DataFrame:
    Qphi_polar = fits.getdata(paths.data / folder / f"{folder}_HD169142_Qphi_polar.fits")
    Uphi_polar = fits.getdata(paths.data / folder / f"{folder}_HD169142_Uphi_polar.fits")

    rin = np.floor(rmin / target_info.dist_pc / pxscales[folder]).astype(int)  # px
    rout = np.ceil(rmax / target_info.dist_pc / pxscales[folder]).astype(int)  # px

    rs = np.arange(Qphi_polar.shape[0])  # px
    mask = (rs >= rin) & (rs <= rout)

    coms = []
    com_errs = []
    for az_idx in range(Qphi_polar.shape[1]):
        Qphi_slice = Qphi_polar[mask, az_idx]
        Uphi_slice = Uphi_polar[mask, az_idx]
        com, com_err = centroid_with_errors(rs[mask], Qphi_slice, Uphi_slice)
        coms.append(com)
        com_errs.append(com_err)

    degs_per_px = 5
    azimuth_deg = np.arange(Qphi_polar.shape[1]) * degs_per_px
    com_au = np.array(coms) * target_info.dist_pc * pxscales[folder]
    com_errs_au = np.array(com_errs) * target_info.dist_pc * pxscales[folder]
    table = pd.DataFrame(
        {
            "azimuth(deg)": azimuth_deg,
            "peak_location(au)": com_au,
            "peak_location_err(au)": com_errs_au,
        }
    )
    return table


if __name__ == "__main__":
    for folder in tqdm.tqdm(folders):
        # inner ring
        table_inner = get_centroids(folder, rmin=15, rmax=35)
        table_inner.insert(1, "region", "inner")
        table_outer = get_centroids(folder, rmin=48, rmax=110)
        table_outer.insert(1, "region", "outer")

        table = pd.concat((table_inner, table_outer))

        filename = paths.data / folder / f"{folder}_HD169142_azimuthal_centroids.csv"
        table.to_csv(filename, index=False)
