import numpy as np
import pandas as pd
import paths
import tqdm
from astropy.io import fits
from target_info import target_info
from utils_errorprop import bootstrap_argmax_and_max
from utils_organization import folders, pxscales


def get_peaks(folder: str, rmin: float, rmax: float) -> pd.DataFrame:
    Qphi_polar = fits.getdata(paths.data / folder / f"{folder}_HD169142_Qphi_polar.fits")
    Uphi_polar = fits.getdata(paths.data / folder / f"{folder}_HD169142_Uphi_polar.fits")

    rin = np.floor(rmin / target_info.dist_pc / pxscales[folder]).astype(int)
    rout = np.ceil(rmax / target_info.dist_pc / pxscales[folder]).astype(int)

    rs = np.arange(Qphi_polar.shape[0])
    mask = (rs >= rin) & (rs <= rout)

    peaks = []
    peak_errs = []
    for az_idx in range(Qphi_polar.shape[1]):
        Qphi_slice = Qphi_polar[mask, az_idx]
        Uphi_slice = Uphi_polar[mask, az_idx]
        peak, peakerr, _, _ = bootstrap_argmax_and_max(rs[mask], Qphi_slice, Uphi_slice, 1000)
        peaks.append(peak)
        peak_errs.append(peakerr)

    degs_per_px = 5
    azimuth_deg = np.arange(Qphi_polar.shape[1]) * degs_per_px
    r_peaks_au = np.array(peaks) * target_info.dist_pc * pxscales[folder]
    r_peak_errs_au = np.array(peak_errs) * target_info.dist_pc * pxscales[folder]
    table = pd.DataFrame(
        {
            "azimuth(deg)": azimuth_deg,
            "peak_location(au)": r_peaks_au,
            "peak_location_err(au)": r_peak_errs_au,
        }
    )
    return table


if __name__ == "__main__":
    for folder in tqdm.tqdm(folders):
        # inner ring
        table_inner = get_peaks(folder, rmin=15, rmax=35)
        table_inner.insert(1, "region", "inner")
        table_outer = get_peaks(folder, rmin=48, rmax=110)
        table_outer.insert(1, "region", "outer")

        table = pd.concat((table_inner, table_outer))

        filename = paths.data / folder / f"{folder}_HD169142_radial_peaks.csv"
        table.to_csv(filename, index=False)
