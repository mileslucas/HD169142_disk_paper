import numpy as np
import pandas as pd
import paths
import proplot as pro
from utils_crosscorr import bootstrap_phase_correlogram
from utils_errorprop import bootstrap_argmax_and_max, bootstrap_interpolate, relative_deviation
from utils_organization import folders, get_time_delta_yr, label_from_folder
from utils_plots import setup_rc

if __name__ == "__main__":
    setup_rc()

    ## Plot and save
    width = 3.31314
    fig, axes = pro.subplots(nrows=1, ncols=2, width=f"{width}in", wspace=0.75, spanx=False)

    labels = [label_from_folder(f) for f in folders]

    curves: dict[str, list] = {"inner": [], "inner_err": [], "outer": [], "outer_err": []}

    for _folder_idx, folder in enumerate(folders):
        # load data
        table = pd.read_csv(paths.data / folder / f"{folder}_HD169142_azimuthal_profiles.csv")

        groups = table.groupby("region")

        for reg_name, group in groups:
            values, errs = relative_deviation(group["Qphi"].values, group["Qphi_err"].values)

            curves[reg_name].append(values)
            curves[f"{reg_name}_err"].append(errs)

    common_lag = np.linspace(-17.5, 17.5, 100)

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
            if dt < 1.2:
                continue
            lags, xcorr, xcorr_err = bootstrap_phase_correlogram(
                curve2, curve2_err, curve1, curve1_err, degs_per_px=5
            )
            lags = lags / dt
            inds = np.argsort(lags)
            extrap, extrap_err = bootstrap_interpolate(
                common_lag, lags[inds], xcorr[inds], xcorr_err[inds]
            )
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
            if dt < 5:
                continue
            lags, xcorr, xcorr_err = bootstrap_phase_correlogram(
                curve2, curve2_err, curve1, curve1_err, degs_per_px=5
            )
            lags = lags / dt
            inds = np.argsort(lags)
            extrap, extrap_err = bootstrap_interpolate(
                common_lag, lags[inds], xcorr[inds], xcorr_err[inds]
            )
            xcorrs_outer.append(extrap)
            xcorrs_outer_err.append(extrap_err)

    mean_xcorr_inner = np.nanmean(xcorrs_inner, axis=0)
    mean_xcorr_inner_std = np.nanstd(xcorrs_inner, axis=0) / np.sqrt(len(xcorrs_inner))
    mean_xcorr_inner_err = np.sqrt(
        np.nansum(np.power(xcorrs_inner_err, 2), axis=0) / len(xcorrs_inner_err) ** 2
        + mean_xcorr_inner_std**2
    )
    norm_val = np.nanmax(mean_xcorr_inner)

    innerpeak, innerpeak_err, _, _ = bootstrap_argmax_and_max(
        common_lag, mean_xcorr_inner, mean_xcorr_inner_err
    )
    with open(paths.data / "cross_correlation_peaks.csv", "w") as fh:
        fh.write(f"inner,{innerpeak},{innerpeak_err}\n")
    print(f"Inner ring peak correlation: {innerpeak} ± {innerpeak_err} deg/yr")

    axes[0].plot(common_lag, mean_xcorr_inner / norm_val, shadedata=mean_xcorr_inner_err / norm_val)
    axes[0].axvline(innerpeak, c="C0", lw=1, alpha=0.7)
    axes[0].format(title="Inner ring")

    mean_xcorr_outer = np.nanmean(xcorrs_outer, axis=0)
    mean_xcorr_outer_std = np.nanstd(xcorrs_outer, axis=0) / np.sqrt(len(xcorrs_outer))
    mean_xcorr_outer_err = np.sqrt(
        np.nansum(np.power(xcorrs_outer_err, 2), axis=0) / len(xcorrs_outer_err) ** 2
        + mean_xcorr_outer_std**2
    )
    norm_val = np.nanmax(mean_xcorr_outer)

    outerpeak, outerpeak_err, _, _ = bootstrap_argmax_and_max(
        common_lag, mean_xcorr_outer, mean_xcorr_outer_err
    )
    print(f"Outer ring peak correlation: {outerpeak} ± {outerpeak_err} deg/yr")

    with open(paths.data / "cross_correlation_peaks.csv", "a") as fh:
        fh.write(f"outer,{outerpeak},{outerpeak_err}\n")

    mask = (common_lag >= -7.5) & (common_lag <= 7.5)
    axes[1].plot(
        common_lag[mask],
        mean_xcorr_outer[mask] / norm_val,
        shadedata=mean_xcorr_outer_err[mask] / norm_val,
        c="C3",
    )
    axes[1].axvline(outerpeak, c="C3", lw=1, alpha=0.7)
    axes[1].format(title="Outer ring")

    for ax in axes:
        ax.axhline(0, c="0.3", lw=1, zorder=0)
        ax.axvline(0, c="0.3", lw=1, zorder=0)

    axes.format(
        ylim=(-0.4, None),
        xlabel="Motion (°/yr)",
        ylabel="Phase cross-correlation",
        yformatter="none",
    )
    axes[0].format(xlocator=10)
    axes[1].format(xlocator=5)

    inner_kep = [-5.18 - 0.18, -5.18 + 0.18]
    axes[0].axvline(-5.18, c="k", alpha=0.9, zorder=-1, lw=1)
    # axes[0].fill_betweenx([ymin, ymax], *inner_kep, c="0.3", alpha=0.2, zorder=-1)

    # add little bars showing spread
    yloc = -0.3
    axes[0].scatter(innerpeak, yloc, c="C0", marker=".", markersize=15, zorder=10)
    axes[0].plot(
        [innerpeak - innerpeak_err, innerpeak + innerpeak_err],
        [yloc, yloc],
        c="C0",
        lw=1,
        zorder=10,
    )
    axes[0].scatter(-5.18, yloc + 0.05, c="k", marker=".", markersize=15, zorder=10)
    axes[0].plot(inner_kep, [yloc + 0.05, yloc + 0.05], c="k", lw=1, zorder=10)

    outer_kep = [-0.926 - 0.083, -0.926 + 0.083]
    axes[1].axvline(-0.926, c="k", alpha=0.9, zorder=-1, lw=1)
    # axes[1].fill_betweenx([ymin, ymax], *outer_kep, c="0.3", alpha=0.2, zorder=-1)

    axes[1].scatter(outerpeak, yloc, c="C3", marker=".", markersize=15, zorder=10)
    axes[1].plot(
        [outerpeak - outerpeak_err, outerpeak + outerpeak_err],
        [yloc, yloc],
        c="C3",
        lw=1,
        zorder=10,
    )
    axes[1].scatter(-0.926, yloc + 0.05, c="k", marker=".", markersize=15, zorder=10)
    axes[1].plot(outer_kep, [yloc + 0.05, yloc + 0.05], c="k", lw=1, zorder=10)

    fig.savefig(paths.figures / "HD169142_azimuthal_profiles_crosscorr.pdf", bbox_inches="tight")
