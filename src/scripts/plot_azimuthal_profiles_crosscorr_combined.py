import proplot as pro
import paths
import pandas as pd
from astropy.time import Time
from scipy import signal
import numpy as np
import scienceplots

def time_from_folder(foldername: str):
    date_raw = foldername.split("_")[0]
    ymd = {
        "year": int(date_raw[:4]),
        "month": int(date_raw[4:6]),
        "day": int(date_raw[6:])
    }
    return Time(ymd, format="ymdhms")

def cross_correlate(curve1, time1: str, curve2, time2: str):
    time_delta = time_from_folder(time1) - time_from_folder(time2)
    time_delta_yr = time_delta.jd / 365.25 # days to years

    fft1 = np.fft.fft(curve1)
    fft2 = np.fft.fft(curve2)

    R = fft1 * fft2.conj()
    r = np.fft.ifft(R)
    cross_corr = np.real(np.fft.fftshift(r))
    # cross_corr /= cross_corr.max()
    # lags_inds = signal.correlation_lags(len(curve1), len(curve2))

    lags_inds = np.arange(-len(curve1)//2, len(curve2)//2)
    # lags_inds = 1 / np.fft.fftshift(np.fft.fftfreq(len(curve1)))
    # bin_width in azimuthal profile is 5 deg
    lags_deg_per_yr = lags_inds * 5 / time_delta_yr
    return lags_deg_per_yr, cross_corr


def label_from_folder(foldername):
    tokens = foldername.split("_")
    date = f"{tokens[0][:4]}/{tokens[0][4:6]}/{tokens[0][6:]}"
    return f"{date} {tokens[1]}"


if __name__ == "__main__":
    pro.rc["font.size"] = 8
    pro.rc["label.size"] = 8
    pro.rc["figure.dpi"] = 300

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

    # colors = [f"C{i}" for i in range(len(folders))]
    color_map = {"inner": "C0", "outer": "C1"}
    ax_map = {"inner": 1, "outer": 0}
    for folder_idx, folder in enumerate(folders):
        # load data
        table = pd.read_csv(
            paths.data / folder / f"{folder}_HD169142_azimuthal_profiles.csv"
        )
        # if "VAMPIRES" in folder:
        #     table = process_vampires(table)

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
            lags, xcorr = cross_correlate(curve2, folder2, curve1,folder1)
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
            lags, xcorr = cross_correlate(curve2, folder2, curve1,folder1)
            inds = np.argsort(lags)
            extrap = np.interp(common_lag, lags[inds], xcorr[inds], left=np.nan, right=np.nan)
            xcorrs_outer.append(extrap)

    mean_xcorr_inner = np.nanmean(xcorrs_inner, axis=0)
    norm_xcorr_inner = mean_xcorr_inner / np.nanmax(mean_xcorr_inner)

    axes[0].plot(common_lag, norm_xcorr_inner)
    max_corr_ind = np.nanargmax(norm_xcorr_inner)
    axes[0].axvline(common_lag[max_corr_ind], c="C0", lw=1, alpha=0.6)
    print(common_lag[max_corr_ind])


    mean_xcorr_outer = np.nanmean(xcorrs_outer, axis=0)
    norm_xcorr_outer = mean_xcorr_outer / np.nanmax(mean_xcorr_outer)

    axes[1].plot(common_lag, norm_xcorr_outer, c="C1")
    max_corr_ind = np.nanargmax(norm_xcorr_outer)
    axes[1].axvline(common_lag[max_corr_ind], c="C1", lw=1, alpha=0.6)
    print(common_lag[max_corr_ind])
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

    # axes[-1].legend(ncols=1, fontsize=8, order="F")


    axes.format(
        xlabel="Lag (°/yr)",
        xlocator=20,
        # ylabel="Normalized cross-correlation",
        # yformatter="none",
        toplabels=("Inner Ring", "Outer Ring")
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
    
