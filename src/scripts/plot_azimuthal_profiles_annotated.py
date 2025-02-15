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
    width = 3.31314
    aspect_ratio = 1 / (3 * 1.61803)
    height = width * aspect_ratio
    fig, axes = pro.subplots(
        nrows=8, width=f"{width}in", refheight=f"{height}in", hspace=0.7
    )

    labels = [label_from_folder(f) for f in folders]

    # colors = [f"C{i}" for i in range(len(folders))]
    for folder_idx, folder in enumerate(folders):

        # load data
        table = pd.read_csv(
            paths.data / folder / f"{folder}_HD169142_azimuthal_profiles.csv"
        )

        sub_table = table.query("region == 'inner'")
        
        values, error = relative_deviation(sub_table["Qphi"], sub_table["Qphi_err"])
        axes[folder_idx].plot(
            sub_table["azimuth(deg)"],
            values * 100,
            shadedata = error * 100,
            c="C0",
        )        
        labels = label_from_folder(folder).split()
        axes[folder_idx].text(
            0.01, 0.95, labels[0], transform="axes", c="0.1 ", ha="left", va="top", fontweight="bold"
        )
        axes[folder_idx].text(
            0.99, 0.95, "\n".join(labels[1:]), transform="axes", c="0.1 ", ha="right", va="top", fontweight="bold"
        )

    for ax in axes:
        ax.axhline(0, c="0.3", lw=1, zorder=0)

        for offset in (90, 270):
            ax.axvline(offset + target_info.pos_angle, c="0.1", lw=1)

    axes.format(
        xlabel="Azimuth (° East of North)",
        ylabel=r"Normalized azimuthal profile $\times r^2$",
        xlocator=90,
        yformatter="percent",
    )

    fig.savefig(
        paths.figures / "HD169142_azimuthal_profiles_inner_annotated.pdf",
        bbox_inches="tight",
    )
