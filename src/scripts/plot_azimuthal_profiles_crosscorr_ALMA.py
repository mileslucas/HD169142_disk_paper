import proplot as pro
import paths
import pandas as pd
from astropy.time import Time
import numpy as np
from utils_crosscorr import bootstrap_phase_correlogram, phase_correlogram
from target_info import target_info
from utils_ephemerides import keplerian_warp
from astropy.io import fits
import tqdm

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
    pro.rc["figure.dpi"] = 300
    pro.rc["font.size"] = 8
    pro.rc["title.size"] = 9
    pro.rc["cycle"] = "ggplot"

    ## Plot and save
    width = 3.31314
    aspect_ratio = 1/1.6
    height = width * aspect_ratio
    fig, axes = pro.subplots(
        width=f"{width}in", height=f"{height}in"
    )

    folders = [
        "20120726_NACO",
        "20140425_GPI",
        "20150503_IRDIS",
        "20150710_ZIMPOL",
        "20180715_ZIMPOL",
        "20210906_IRDIS",
        # "20230707_VAMPIRES",
        "20240729_VAMPIRES",
    ]
    labels = [label_from_folder(f) for f in folders]
    pxscales = {
        "20120726_NACO": 27e-3,
        "20140425_GPI": 14.14e-3,
        "20150503_IRDIS": 12.25e-3,
        "20150710_ZIMPOL": 3.6e-3,
        "20170918_ALMA": 5e-3,
        "20180715_ZIMPOL": 3.6e-3,
        "20230707_VAMPIRES": 5.9e-3,
        "20210906_IRDIS": 12.25e-3,
        "20240729_VAMPIRES": 5.9e-3,
    }

    alma_folder = "20170918_ALMA"
    alma_data = fits.getdata(paths.data / alma_folder / f"{alma_folder}_HD169142_Qphi_polar.fits")
    rs = np.arange(alma_data.shape[0])
    rin = np.floor(15 / target_info.dist_pc / pxscales[alma_folder]).astype(int)
    rout = np.ceil(35 / target_info.dist_pc / pxscales[alma_folder]).astype(int)
    mask = (rs >= rin) & (rs <= rout)
    alma_curve = np.nanmean(alma_data[mask, :], axis=0)
    alma_curve = alma_curve / alma_curve.mean() - 1
    alma_err = np.nanstd(alma_data[mask, :], axis=0)

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
        polar_cube_rolled = keplerian_warp(polar_cube[mask, :], rs_au, this_time, alma_time)

        this_curve = np.nanmean(polar_cube_rolled, axis=0)
        this_curve_err = np.nanstd(polar_cube_rolled, axis=0)
        this_curve = this_curve / this_curve.mean() - 1

        dt_yr = (this_time - alma_time).jd / 365.25
        lags_degs, xcorr = phase_correlogram(alma_curve, this_curve)
        # lags, xcorr, xcorr_err = bootstrap_phase_correlogram(curve2.values, curve2_err.values, curve1.values, curve1_err.values, N=1000)

        xcorr_itp = np.interp(common_lags, lags_degs, xcorr)
        xcorrs.append(xcorr_itp)


    mean_xcorr = np.mean(xcorrs, axis=0)
    peak_idx = np.argmax(mean_xcorr)
    best_lag = common_lags[peak_idx]
    for xcorr in xcorrs:
        axes[0].plot(common_lags, xcorr, c="C3", alpha=0.6, zorder=5, lw=0.8)
    axes[0].plot(common_lags, mean_xcorr, c="C0", zorder=10)
    axes[0].axvline(best_lag, c="C0", alpha=0.8, zorder=2, lw=1)
    print(f"Peak: {best_lag} (deg)")


    for ax in axes:
        ax.axhline(0, c="0.3", lw=1, zorder=0)
        ax.axvline(0, c="0.3", lw=1, zorder=0)

    # axes[-1].legend(ncols=1, fontsize=8, order="F")

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
    
