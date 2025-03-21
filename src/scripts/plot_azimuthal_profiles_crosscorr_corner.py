import proplot as pro
import paths
import pandas as pd
from astropy.time import Time
from scipy import signal
import itertools
import numpy as np
from utils_crosscorr import bootstrap_phase_correlogram, phase_correlogram
from utils_errorprop import relative_deviation
from utils_organization import get_time_delta_yr, label_from_folder, folders
from utils_plots import setup_rc

if __name__ == "__main__":
    setup_rc()
    ## Plot and save
    fig, axes = pro.subplots(
        nrows=len(folders) - 1, ncols=len(folders) - 1, width="7in", space=1
    )

    labels = [label_from_folder(f) for f in folders]

    curves: dict[str, list] = {"inner": [], "inner_err": [], "outer": [], "outer_err": []}

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
            # need to mean subtract data to remove DC power term from crosscorr
            # by making it relative we also correct for differences in units
            values, errs = relative_deviation(group["Qphi"].values, group["Qphi_err"].values)

            curves[reg_name].append(values)
            curves[f"{reg_name}_err"].append(errs)

    # combs_inner = list(itertools.combinations(curves["inner"], 2))
    for col_idx in range(len(folders) - 1):
        curve1 = curves["inner"][col_idx]
        curve1_err = curves["inner_err"][col_idx]
        folder1 = folders[col_idx]
        for row_idx in range(col_idx, len(folders)):
            if row_idx == col_idx:
                continue
            curve2 = curves["inner"][row_idx]
            curve2_err = curves["inner_err"][row_idx]
            folder2 = folders[row_idx]
            dt_yr = get_time_delta_yr(folder1, folder2)
            # if dt_yr < 1.2:
            #     continue
            lags, xcorr, xcorr_err = bootstrap_phase_correlogram(curve2, curve2_err, curve1, curve1_err, degs_per_px=5)
            lags_degs_per_yr = lags / dt_yr
            axes[row_idx - 1, col_idx].plot(lags_degs_per_yr, xcorr, shadedata=xcorr_err, c="C0")
    
            if dt_yr < 1.2:
                axes[row_idx - 1, col_idx].text(
                    0.97, 0.97, "(*)",
                    c="0.3",
                    transform="axes",
                    fontweight="bold",
                    ha="right",
                    va="top",
                )
    for ax in axes:
        ax.axhline(0, c="0.3", lw=1, zorder=0)
        ax.axvline(0, c="0.3", lw=1, zorder=0)


    for row_idx in range(7):
        for col_idx in range(row_idx + 1, 7):
            axes[row_idx, col_idx].set_visible(False)


    axes.format(
        xlim=(-50, 50),
        xlocator=20,
        xlabel="Lag (°/yr)",
        yformatter="none",
        leftlabels=("2014 GPI", "2015 IRDIS", "2015 ZIMPOL", "2018 ZIMPOL", "2021 IRDIS", "2023 CHARIS", "2024 VAMPIRES"),
    )
    toplabels=("2012 NACO", "2014 GPI", "2015 IRDIS", "2015 ZIMPOL", "2018 ZIMPOL", "2021 IRDIS", "2023 CHARIS")

    for idx in range(len(folders) - 1):
        axes[idx, idx].format(title=toplabels[idx], titleweight="bold")


    fig.savefig(
        paths.figures / "HD169142_azimuthal_profiles_crosscorr_corner_inner.pdf",
        bbox_inches="tight",
    )


    ## 2
    
    fig, axes = pro.subplots(
        nrows=len(folders) - 1, ncols=len(folders) - 1, width="7in", space=1
    )
    # combs_inner = list(itertools.combinations(curves["inner"], 2))
    for col_idx in range(len(folders) - 1):
        curve1 = curves["outer"][col_idx]
        curve1_err = curves["outer_err"][col_idx]
        folder1 = folders[col_idx]
        for row_idx in range(col_idx, len(folders)):
            if row_idx == col_idx:
                continue
            curve2 = curves["outer"][row_idx]
            curve2_err = curves["outer_err"][row_idx]
            folder2 = folders[row_idx]
            dt_yr = get_time_delta_yr(folder1, folder2)
            lags, xcorr, xcorr_err = bootstrap_phase_correlogram(curve2, curve2_err, curve1, curve1_err, degs_per_px=5)
            lags_degs_per_yr = lags / dt_yr
            axes[row_idx - 1, col_idx].plot(lags_degs_per_yr, xcorr, shadedata=xcorr_err, c="C3")
    
            if dt_yr < 5 or "CHARIS" in folder1 or "CHARIS" in folder2:
                axes[row_idx - 1, col_idx].text(
                    0.97, 0.97, "(*)",
                    c="0.3",
                    transform="axes",
                    fontweight="bold",
                    ha="right",
                    va="top",
                )


    for ax in axes:
        ax.axhline(0, c="0.3", lw=1, zorder=0)
        ax.axvline(0, c="0.3", lw=1, zorder=0)


    for row_idx in range(7):
        for col_idx in range(row_idx + 1, 7):
            axes[row_idx, col_idx].set_visible(False)


    axes.format(
        xlim=(-50, 50),
        xlocator=20,
        xlabel="Lag (°/yr)",
        yformatter="none",
        leftlabels=("2014 GPI", "2015 IRDIS", "2015 ZIMPOL", "2018 ZIMPOL", "2021 IRDIS", "2023 CHARIS", "2024 VAMPIRES"),
    )
    toplabels=("2012 NACO", "2014 GPI", "2015 IRDIS", "2015 ZIMPOL", "2018 ZIMPOL", "2021 IRDIS", "2023 CHARIS")

    for idx in range(len(folders) - 1):
        axes[idx, idx].format(title=toplabels[idx], titleweight="bold")


    fig.savefig(
        paths.figures / "HD169142_azimuthal_profiles_crosscorr_corner_outer.pdf",
        bbox_inches="tight",
    )