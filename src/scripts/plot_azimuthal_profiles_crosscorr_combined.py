import proplot as pro
import paths
import pandas as pd
from astropy.time import Time
from scipy import signal
import numpy as np
from utils_crosscorr import phase_correlogram

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

def get_time_delta_yr(folder1: str, folder2: str) -> float:
    time1 = time_from_folder(folder1)
    time2 = time_from_folder(folder2)
    return (time2 - time1).jd / 365.25

if __name__ == "__main__":
    pro.rc["font.size"] = 8
    pro.rc["title.size"] = 9
    pro.rc["label.size"] = 8
    pro.rc["figure.dpi"] = 300
    pro.rc["cycle"] = "ggplot"

    ## Plot and save
    width = 3.31314
    aspect_ratio = 0.66
    height = width * aspect_ratio
    fig, axes = pro.subplots(
        nrows=1, ncols=2, width=f"{width}in", wspace=0.75, spanx=False
    )

    folders = [
        "20120726_NACO",
        "20140425_GPI",
        "20150503_IRDIS",
        "20150710_ZIMPOL",
        "20180715_ZIMPOL",
        "20210906_IRDIS",
        # "20230707_VAMPIRES",
        # "20240729_VAMPIRES",
    ]
    labels = [label_from_folder(f) for f in folders]

    curves: dict[str, list] = {"inner": [], "outer": []}

    for folder_idx, folder in enumerate(folders):
        # load data
        table = pd.read_csv(
            paths.data / folder / f"{folder}_HD169142_azimuthal_profiles.csv"
        )

        groups = table.groupby("region")
        
        for reg_name, group in groups:
            values, errs = group["Qphi"], group["Qphi_err"]

            values = (values / values.mean() - 1)
            curves[reg_name].append(values)


    common_lag = np.linspace(-50, 50, 100)

    xcorrs_inner = []
    xcorrs_outer = []

    for col_idx in range(len(folders) - 1):
        curve1 = curves["inner"][col_idx]
        folder1 = folders[col_idx]
        for row_idx in range(col_idx, len(folders)):
            if row_idx == col_idx:
                continue
            curve2 = curves["inner"][row_idx]
            folder2 = folders[row_idx]
            dt = get_time_delta_yr(folder1, folder2)
            lags, xcorr = phase_correlogram(curve2, curve1)
            lags = lags / dt
            inds = np.argsort(lags)
            extrap = np.interp(common_lag, lags[inds], xcorr[inds], left=np.nan, right=np.nan)
            xcorrs_inner.append(extrap)
        curve1 = curves["outer"][col_idx]
        folder1 = folders[col_idx]
        for row_idx in range(col_idx, len(folders)):
            if row_idx == col_idx:
                continue
            curve2 = curves["outer"][row_idx]
            folder2 = folders[row_idx]
            dt = get_time_delta_yr(folder1, folder2)
            lags, xcorr = phase_correlogram(curve2, curve1)
            lags = lags / dt
            inds = np.argsort(lags)
            extrap = np.interp(common_lag, lags[inds], xcorr[inds], left=np.nan, right=np.nan)
            xcorrs_outer.append(extrap)

    mean_xcorr_inner = np.nanmean(xcorrs_inner, axis=0)
    norm_xcorr_inner = mean_xcorr_inner / np.nanmax(mean_xcorr_inner)

    axes[0].plot(common_lag, norm_xcorr_inner)
    max_corr_ind = np.nanargmax(norm_xcorr_inner)
    axes[0].axvline(common_lag[max_corr_ind], c="C0", lw=1, alpha=0.6)
    print(f"Inner ring peak correaltion: {common_lag[max_corr_ind]} deg/yr")
    axes[0].format(title="Inner ring")

    mean_xcorr_outer = np.nanmean(xcorrs_outer, axis=0)
    norm_xcorr_outer = mean_xcorr_outer / np.nanmax(mean_xcorr_outer)

    axes[1].plot(common_lag, norm_xcorr_outer, c="C3")
    max_corr_ind = np.nanargmax(norm_xcorr_outer)
    axes[1].axvline(common_lag[max_corr_ind], c="C3", lw=1, alpha=0.6)
    axes[1].format(title="Outer ring")
    print(f"Outer ring peak correaltion: {common_lag[max_corr_ind]} deg/yr")
    # axes[1, 0].text(
    #     0.03,
    #     0.95,
    #     "Inner",
    #     c="0.3",
    #     fontsize=9,
    #     transform="axes",
    #     ha="left",
    #     va="top",
    # )
    # axes[0, 0].text(
    #     0.03,
    #     0.95,
    #     "Outer",
    #     c="0.3",
    #     fontsize=9,
    #     transform="axes",
    #     ha="left",
    #     va="top",
    # )

    for ax in axes:
        ax.axhline(0, c="0.3", lw=1, zorder=0)
        ax.axvline(0, c="0.3", lw=1, zorder=0)

    axes.format(
        xlabel="Lag (°/yr)",
        ylabel="Phase cross-correlation",
        yformatter="none",
        xlocator=20,
    )

    ymin, ymax = axes[0].get_ylim()
    inner_kep = [-7.96, -2.23]
    axes[0].fill_betweenx([ymin, ymax], *inner_kep, c="0.3", alpha=0.2, zorder=-1)

    ymin, ymax = axes[1].get_ylim()
    outer_kep = [-1.39, -0.4]
    axes[1].fill_betweenx([ymin, ymax], *outer_kep, c="0.3", alpha=0.2, zorder=-1)

    fig.savefig(
        paths.figures / "HD169142_azimuthal_profiles_crosscorr_combined.pdf",
        bbox_inches="tight",
    )


    ## 2
    
