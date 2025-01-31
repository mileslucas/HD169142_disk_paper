import proplot as pro
import paths
import numpy as np
import pandas as pd
from target_info import target_info

## Plot and save


def label_from_folder(foldername):
    tokens = foldername.split("_")
    date = f"{tokens[0][:4]}/{tokens[0][4:6]}/{tokens[0][6:]}"
    return f"{date} {tokens[1]}"


if __name__ == "__main__":
    pro.rc["font.size"] = 7
    pro.rc["title.size"] = 8
    pro.rc["cycle"] = "ggplot"

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
    ## Plot and save
    fig, axes = pro.subplots(
        nrows=2, ncols=4, width="6.66in", refheight="0.8in", wspace=0, hspace=0.75
    )

    labels = [label_from_folder(f) for f in folders]

    # colors = [f"C{i}" for i in range(len(folders))]
    for folder_idx, folder in enumerate(folders):
        axes_row = folder_idx // 4
        axes_col = folder_idx % 4
        # load data
        table = pd.read_csv(
            paths.data / folder / f"{folder}_HD169142_radial_profiles.csv"
        )
        # if "VAMPIRES" in folder:
        #     table = process_vampires(table)

        values, errs = table["Qphi"], table["Qphi_err"]
        mask = table["radius(au)"] <= 120
        norm_val = values.loc[mask].max()
        axes[axes_row, axes_col].plot(
            table["radius(au)"],
            values / norm_val,
            zorder=100,
            lw=1
        )
        axes[axes_row, axes_col].text(
            0.95, 0.97,
            labels[folder_idx],
            c="C0",
            fontsize=6,
            fontweight="bold",
            transform="axes",
            ha="right",
            va="top",
        )

    alma_table = pd.read_csv(paths.data / "20170918_ALMA" / "20170918_ALMA_HD169142_radial_profiles.csv")

    for ax in axes:
        # baseline
        values, errs = alma_table["I"], alma_table["I_err"]
        norm_val = values.max()
        ax.plot(
            alma_table["radius(au)"],
            values / norm_val,
            c="C3",
            lw=1,
            zorder=1
        )
        ax.text(
            0.95, 0.89,
            "ALMA",
            c="C3",
            fontsize=6,
            fontweight="bold",
            transform="axes",
            ha="right",
            va="top",
        )
        

    # axes[-1].legend(ncols=1, fontsize=8, order="F")

    axes.format(
        xlim=(0, 115),
        ylim=(-0.05, 1.1),
        xlabel="Separation (au)",
        ylabel=r"Normalized radial profile $\times r^2$",
        # ylocator=0.25,
    )
    # axes[1:].format(yspineloc="none")

    fig.savefig(
        paths.figures / "HD169142_radial_profiles_combo.pdf",
        bbox_inches="tight",
        dpi=300,
    )
