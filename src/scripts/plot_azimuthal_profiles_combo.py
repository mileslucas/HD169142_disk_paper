import proplot as pro
import paths
import numpy as np
import pandas as pd
from target_info import target_info
from utils_errorprop import relative_deviation
from utils_organization import folders, label_from_folder
from utils_plots import setup_rc


if __name__ == "__main__":
    setup_rc()

    ## Plot and save
    fig, axes = pro.subplots(
        nrows=4, ncols=len(folders)//2, width="7in", height=f"{7 / (1.25 * 1.61803)}in", wspace=0.75, hspace=(0, 2.25, 0)
    )

    labels = [label_from_folder(f) for f in folders]

    # colors = [f"C{i}" for i in range(len(folders))]
    color_map = {"inner": "C0", "outer": "C3"}
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
            if folder_idx < 4:
                axes_row = ax_map[reg_name]
                axes_col = folder_idx
            else:
                axes_row = 2 + ax_map[reg_name]
                axes_col = folder_idx % 4
            values, error = relative_deviation(group["Qphi"], group["Qphi_err"])
            axes[axes_row, axes_col].plot(
                group["azimuth(deg)"],
                values * 100,
                shadedata = error * 100,
                c=color_map[reg_name],
            )

        
        labels = label_from_folder(folder).split()
        axes[axes_row, axes_col].text(
            0.03, 1.01, labels[0],
            transform="axes",
            c="0.3",
            fontweight="bold",
            ha="left",
            va="bottom"
        )
        axes[axes_row, axes_col].text(
            0.99, 1.01, " ".join(labels[1:]),
            transform="axes",
            c="0.3",
            fontweight="bold",
            ha="right",
            va="bottom"
        )


    for i in (0, 2):
        axes[i, 0].text(
            0.02,
            0.95,
            "Outer",
            c="C3",
            fontweight="bold",
            transform="axes",
            ha="left",
            va="top",
        )
        axes[i + 1, 0].text(
            0.02,
            0.95,
            "Inner",
            c="C0",
            fontweight="bold",
            transform="axes",
            ha="left",
            va="top",
        )

    for ax in axes:
        # baseline
        ax.axhline(0, c="0.3", lw=1, zorder=0)
        # disk markers
        # norm_pa = np.mod(target_info.pos_angle - 90, 360)
        # ax.axvline(norm_pa, c="0.5", lw=0.5, zorder=0)
        # ax.axvline(norm_pa - 180, c="0.5", lw=0.5, zorder=0)

    axes.format(
        xlabel="Azimuth (° East of North)",
        ylabel=r"Normalized azimuthal profile $\times r^2$",
        xlocator=90,
        yformatter="percent",
        # ylocator=0.25,
    )
    # axes[1:].format(yspineloc="none")

    fig.savefig(
        paths.figures / "HD169142_azimuthal_profiles_combo.pdf",
        bbox_inches="tight",
        dpi=300,
    )
