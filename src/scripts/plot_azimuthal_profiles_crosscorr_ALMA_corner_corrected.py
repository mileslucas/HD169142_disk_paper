import proplot as pro
import paths
import pandas as pd
from astropy.time import Time
from scipy import signal
import itertools
from astropy.io import fits
import numpy as np
from utils_crosscorr import bootstrap_phase_correlogram, phase_correlogram
from target_info import target_info

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
    fig, axes = pro.subplots(
        nrows=8, refheight="1in", refwidth="3.33in", hspace=0.5, #width="7in", height="2.5in",
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
    alma_table = pd.read_csv(paths.data / alma_folder / f"{alma_folder}_HD169142_azimuthal_profiles.csv")
    alma_group = alma_table.groupby("region").get_group("inner")
    alma_curve = (alma_group["I"])
    alma_time = time_from_folder(alma_folder)


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
    for idx in range(len(folders)):
        curve1 = curves["inner"][idx]
        curve1_err = curves["inner_err"][idx]
        folder1 = folders[idx]
        time1 = time_from_folder(folder1)

        dt_yr = (time1 - alma_time).jd / 365.25
        lags, xcorr = phase_correlogram(curve1.values, alma_curve.values)
        # lags, xcorr, xcorr_err = bootstrap_phase_correlogram(curve2.values, curve2_err.values, curve1.values, curve1_err.values, N=1000)
        kep_motion = -5.17 # deg / yr
        lags_degs_per_yr = lags - kep_motion * dt_yr
        inds = np.argsort(lags_degs_per_yr)
        axes[idx].plot(lags_degs_per_yr[inds], xcorr[inds], c="C0")
        labels = label_from_folder(folder1).split()
        axes[idx].text(
            0.01, 0.95, labels[0], transform="axes", c="0.2", ha="left", va="top", fontsize=8, fontweight="bold"
        )
        axes[idx].text(
            0.99, 0.95, labels[1], transform="axes", c="0.2", ha="right", va="top", fontsize=8, fontweight="bold"
        )


    for ax in axes:
        ax.axhline(0, c="0.3", lw=1, zorder=0)
        ax.axvline(0, c="0.3", lw=1, zorder=0)

    # axes[-1].legend(ncols=1, fontsize=8, order="F")

    axes.format(
        xlim=(-90, 90),
        xlabel="Offset (°)",
        yformatter="none",
    )

    fig.savefig(
        paths.figures / "HD169142_azimuthal_profiles_crosscorr_alma_corner_corrected.pdf",
        bbox_inches="tight",
    )


    ## 2
    
