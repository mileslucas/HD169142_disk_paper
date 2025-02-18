import proplot as pro
import paths
import pandas as pd
from astropy.time import Time
import numpy as np
from utils_crosscorr import bootstrap_phase_correlogram
from utils_errorprop import relative_deviation, bootstrap_argmax_and_max
from utils_organization import folders, get_time_delta_yr, label_from_folder
from utils_plots import setup_rc

if __name__ == "__main__":
    setup_rc()

    ## Plot and save
    width = 3.31314
    aspect_ratio = 1.61803
    height = width / aspect_ratio
    fig, axes = pro.subplots(width=f"{width}in", height=f"{height}in")

    labels = [label_from_folder(f) for f in folders]

    curves: dict[str, list] = {"inner": [], "inner_err": [], "outer": [], "outer_err": []}

    for folder_idx, folder in enumerate(folders):
        # load data
        table = pd.read_csv(
            paths.data / folder / f"{folder}_HD169142_azimuthal_profiles.csv"
        )

        groups = table.groupby("region")

        for reg_name, group in groups:
            values, errs = relative_deviation(group["Qphi"].values, group["Qphi_err"].values)

            curves[reg_name].append(values)
            curves[f"{reg_name}_err"].append(errs)


    common_lag = np.linspace(-50, 50, 100)

    xcorrs_inner = []
    xcorrs_inner_err = []
    xcorrs_outer = []
    xcorrs_outer_err = []

    for col_idx in range(len(folders) - 1):
        # INNer
        curve1 = curves["inner"][col_idx]
        curve1_err = curves["inner_err"][col_idx]
        folder1 = folders[col_idx]
        for row_idx in range(col_idx, len(folders)):
            if row_idx == col_idx:
                continue
            curve2 = curves["inner"][row_idx]
            curve2_err = curves["inner_err"][row_idx]
            folder2 = folders[row_idx]
            dt = get_time_delta_yr(folder1, folder2)
            lags, xcorr, xcorr_err = bootstrap_phase_correlogram(curve2, curve2_err, curve1, curve1_err)
            lags = lags / dt
            inds = np.argsort(lags)
            extrap = np.interp(common_lag, lags[inds], xcorr[inds], left=np.nan, right=np.nan)
            extrap_err = np.interp(common_lag, lags[inds], xcorr_err[inds], left=np.nan, right=np.nan)
            xcorrs_inner.append(extrap)
            xcorrs_inner_err.append(extrap_err)

        # outer
        curve1 = curves["outer"][col_idx]
        curve1_err = curves["outer_err"][col_idx]
        folder1 = folders[col_idx]
        for row_idx in range(col_idx, len(folders)):
            if row_idx == col_idx:
                continue
            curve2 = curves["outer"][row_idx]
            curve2_err = curves["outer_err"][row_idx]
            folder2 = folders[row_idx]
            dt = get_time_delta_yr(folder1, folder2)
            lags, xcorr, xcorr_err = bootstrap_phase_correlogram(curve2, curve2_err, curve1, curve1_err)
            lags = lags / dt
            inds = np.argsort(lags)
            extrap = np.interp(common_lag, lags[inds], xcorr[inds], left=np.nan, right=np.nan)
            extrap_err = np.interp(common_lag, lags[inds], xcorr_err[inds], left=np.nan, right=np.nan)
            xcorrs_outer.append(extrap)
            xcorrs_outer_err.append(extrap_err)

    mean_xcorr_inner = np.nanmean(xcorrs_inner, axis=0)
    mean_xcorr_inner_std = np.nanstd(xcorrs_inner, axis=0) / np.sqrt(len(xcorrs_inner))
    mean_xcorr_inner_err = np.sqrt(np.nansum(np.power(xcorrs_inner_err, 2), axis=0) / len(xcorrs_inner_err)**2 + mean_xcorr_inner_std**2)
    norm_val = np.nanmax(mean_xcorr_inner)

    x0, x0err, _, _ = bootstrap_argmax_and_max(common_lag, mean_xcorr_inner, mean_xcorr_inner_err)
    with open(paths.data / "cross_correlation_peaks.csv", "w") as fh:
        fh.write(f"inner,{x0},{x0err}\n")
    print(f"Inner ring peak correlation: {x0} ± {x0err} deg/yr")

    axes[0].plot(common_lag, mean_xcorr_inner/norm_val, shadedata=mean_xcorr_inner_err/norm_val)
    axes[0].axvline(x0, c="C0", lw=1)
    axes[0].format(title="Inner ring")

    mean_xcorr_outer = np.nanmean(xcorrs_outer, axis=0)
    mean_xcorr_outer_std = np.nanstd(xcorrs_outer, axis=0) / np.sqrt(len(xcorrs_outer))
    mean_xcorr_outer_err = np.sqrt(np.nansum(np.power(xcorrs_outer_err, 2), axis=0) / len(xcorrs_outer_err)**2 + mean_xcorr_outer_std**2)
    norm_val = np.nanmax(mean_xcorr_outer)


    x0, x0err, _, _ = bootstrap_argmax_and_max(common_lag, mean_xcorr_outer, mean_xcorr_outer_err)
    print(f"Outer ring peak correlation: {x0} ± {x0err} deg/yr")
    
    with open(paths.data / "cross_correlation_peaks.csv", "a") as fh:
        fh.write(f"outer,{x0},{x0err}\n")

    axes[1].plot(common_lag, mean_xcorr_outer/norm_val, shadedata=mean_xcorr_outer_err/norm_val, c="C3")
    axes[1].axvline(x0, c="C3", lw=1)
    axes[1].format(title="Outer ring")


    for ax in axes:
        ax.axhline(0, c="0.3", lw=1, zorder=0)
        ax.axvline(0, c="0.3", lw=1, zorder=0)

    axes.format(
        xlabel="Motion (°/yr)",
        ylabel="Phase cross-correlation",
        yformatter="none",
        xlocator=20,
    )

    ymin, ymax = axes[0].get_ylim()
    # inner_kep = [-7.96, -2.23]
    inner_kep = [-2.46, -8.76]
    axes[0].fill_betweenx([ymin, ymax], *inner_kep, c="0.3", alpha=0.2, zorder=-1)

    ymin, ymax = axes[1].get_ylim()
    # outer_kep = [-1.39, -0.4]
    outer_kep = [--0.44, -1.53]
    axes[1].fill_betweenx([ymin, ymax], *outer_kep, c="0.3", alpha=0.2, zorder=-1)

    fig.savefig(
        paths.figures / "HD169142_azimuthal_profiles_crosscorr.pdf",
        bbox_inches="tight",
    )


