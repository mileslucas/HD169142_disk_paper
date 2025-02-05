import proplot as pro
import paths
import pandas as pd
from astropy.time import Time
from scipy import signal
import itertools
import numpy as np
from utils_crosscorr import bootstrap_phase_correlogram, phase_correlogram


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
    pro.rc["font.size"] = 9

    ## Plot and save
    fig, axes = pro.subplots(
        nrows=7, ncols=7,# space=0, #width="7in", height="2.5in",
    )

    folders = [
        "20120726_NACO",
        "20140425_GPI",
        "20150503_IRDIS",
        "20150710_ZIMPOL",
        "20180715_ZIMPOL",
        "20210906_IRDIS",
        "20230707_VAMPIRES",
        "20240729_VAMPIRES",
    ]
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
            values, errs = group["Qphi"], group["Qphi_err"]

            values = (values / values.mean() - 1)
            curves[reg_name].append(values)
            curves[f"{reg_name}_err"].append(errs / values.mean())

    # combs_inner = list(itertools.combinations(curves["inner"], 2))
    for col_idx in range(len(folders) - 1):
        curve1 = curves["inner"][col_idx]
        curve1_err = curves["inner_err"][col_idx]
        folder1 = folders[col_idx]
        time1 = time_from_folder(folder1)
        for row_idx in range(col_idx, len(folders)):
            if row_idx == col_idx:
                continue
            curve2 = curves["inner"][row_idx]
            curve2_err = curves["inner_err"][row_idx]
            folder2 = folders[row_idx]
            time2 = time_from_folder(folder2)
            dt_yr = (time2 - time1).jd / 365.25
            lags, xcorr = phase_correlogram(curve2.values, curve1.values)
            # lags, xcorr, xcorr_err = bootstrap_phase_correlogram(curve2.values, curve2_err.values, curve1.values, curve1_err.values, N=1000)
            lags_degs_per_yr = lags / dt_yr
            axes[row_idx - 1, col_idx].plot(lags_degs_per_yr, xcorr, c="C0")
    

    # for col_idx in range(len(folders) - 1):
    #     curve1 = curves["outer"][col_idx]
    #     folder1 = folders[col_idx]
    #     for row_idx in range(col_idx, len(folders)):
    #         if row_idx == col_idx:
    #             continue
    #         curve2 = curves["outer"][row_idx]
    #         folder2 = folders[row_idx]
    #         lags, xcorr = cross_correlate(curve1,folder1, curve2, folder2)
    #         axes[row_idx - 1, col_idx].plot(lags, xcorr, c="C1")

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

    # axes[-1].legend(ncols=1, fontsize=8, order="F")


    for row_idx in range(7):
        for col_idx in range(row_idx + 1, 7):
            axes[row_idx, col_idx].set_visible(False)


    axes.format(
        xlim=(-60, 60),
        xlabel="Lag (°/yr)",
        yformatter="none",
        leftlabels=("2014 GPI", "2015 IRDIS", "2015 ZIMPOL", "2018 ZIMPOL", "2021 IRDIS", "2023 VAMPIRES", "2024 VAMPIRES"),
    )
    toplabels=("2012 NACO", "2014 GPI", "2015 IRDIS", "2015 ZIMPOL", "2018 ZIMPOL", "2021 IRDIS", "2023 VAMPIRES")

    for idx in range(len(folders) - 1):
        axes[idx, idx].format(title=toplabels[idx], titleweight="bold")


    fig.savefig(
        paths.figures / "HD169142_azimuthal_profiles_crosscorr.pdf",
        bbox_inches="tight",
        dpi=300,
    )


    ## 2
    
