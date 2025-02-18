import proplot as pro
import paths
import pandas as pd
from astropy.time import Time
import numpy as np
from utils_crosscorr import bootstrap_phase_correlogram
from target_info import target_info
from utils_ephemerides import keplerian_warp
from astropy.io import fits
from utils_errorprop import relative_deviation, bootstrap_argmax_and_max
import tqdm
from utils_plots import setup_rc
from utils_organization import folders, pxscales

def time_from_folder(foldername: str):
    date_raw = foldername.split("_")[0]
    ymd = {
        "year": int(date_raw[:4]),
        "month": int(date_raw[4:6]),
        "day": int(date_raw[6:])
    }
    return Time(ymd, format="ymdhms")

def label_from_folder(foldername):
    tokens = foldername.split("_")
    date = f"{tokens[0][:4]}/{tokens[0][4:6]}/{tokens[0][6:]}"
    return f"{date} {tokens[1]}"


if __name__ == "__main__":
    setup_rc()

    ## Plot and save
    width = 3.31314
    aspect_ratio = 1/1.6
    height = width * aspect_ratio
    fig, axes = pro.subplots(
        width=f"{width}in", height=f"{height}in"
    )


    alma_folder = "20170918_ALMA_1.3mm"
    alma_polar_frame = fits.getdata(paths.data / alma_folder / f"{alma_folder}_HD169142_Qphi_polar.fits")
    rs = np.arange(alma_polar_frame.shape[0])
    rin = np.floor(15 / target_info.dist_pc / pxscales[alma_folder]).astype(int)
    rout = np.ceil(35 / target_info.dist_pc / pxscales[alma_folder]).astype(int)
    mask = (rs >= rin) & (rs <= rout)

    alma_timestamp = time_from_folder(alma_folder)
    alma_prof = np.mean(alma_polar_frame[mask, :], axis=0)
    alma_prof_std = np.std(alma_polar_frame[mask, :], axis=0) / np.sqrt((rout - rin))
    alma_prof_error = np.hypot(14.7e-3 / np.sqrt((rout - rin)), alma_prof_std)
    alma_norm_prof, alma_norm_err = relative_deviation(alma_prof, alma_prof_error)
    alma_time = time_from_folder(alma_folder)

    # combs_inner = list(itertools.combinations(curves["inner"], 2))
    common_lags = np.arange(-90, 90)
    xcorrs = []
    errs = []

    for idx, folder in enumerate(tqdm.tqdm(folders)):
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

        this_curve = np.mean(polar_cube_warped, axis=0)
        this_curve_err = np.std(polar_cube_warped, axis=0) / np.sqrt(polar_cube_warped.shape[0])
        norm_curve, norm_err = relative_deviation(this_curve, this_curve_err)

        lags_degs, xcorr, xcorr_err = bootstrap_phase_correlogram(alma_norm_prof, alma_norm_err, norm_curve, norm_err)

        xcorr_itp = np.interp(common_lags, lags_degs, xcorr)
        xcorrs.append(xcorr_itp)
        
        xcorr_err_itp = np.interp(common_lags, lags_degs, xcorr_err)
        errs.append(xcorr_err_itp)


    mean_xcorr = np.mean(xcorrs, axis=0)
    stderr_xcorr = np.std(xcorrs, axis=0) / np.sqrt(len(xcorrs))
    rmserr_xcorr = np.sqrt(np.sum(np.power(errs, 2), axis=0)) / len(errs)
    err_xcorr = np.hypot(stderr_xcorr, rmserr_xcorr)

    best_lag, best_lag_err, _, _ = bootstrap_argmax_and_max(common_lags, mean_xcorr, err_xcorr)
    # for i in range(len(xcorrs)):
    #     axes[0].plot(common_lags, xcorrs[i], shadedata=errs[i], c="C3", alpha=0.6, zorder=5, lw=0.8)
    axes[0].plot(common_lags, mean_xcorr, shadedata=err_xcorr, c="C0", zorder=10)
    axes[0].axvline(best_lag, c="C0", alpha=0.8, zorder=2, lw=1)
    print(f"Peak: {best_lag} ± {best_lag_err} (deg)")


    for ax in axes:
        ax.axhline(0, c="0.3", lw=1, zorder=0)
        ax.axvline(0, c="0.3", lw=1, zorder=0)


    axes.format(
        xlim=(-90, 90),
        xlabel="Offset (°)",
        ylabel="Phase cross-correlation",
        xlocator=20,
        yformatter="none",
    )

    fig.savefig(
        paths.figures / "HD169142_azimuthal_profiles_ALMA_crosscorr.pdf",
        bbox_inches="tight",
    )


    ## 2
    
