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
        nrows=4, ncols=len(folders)//2, width="7in", height="4in", wspace=0.5, hspace=(0, 3, 0)
    )

    labels = [label_from_folder(f) for f in folders]

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
            if folder_idx < 4:
                axes_row = ax_map[reg_name]
                axes_col = folder_idx
            else:
                axes_row = 2 + ax_map[reg_name]
                axes_col = folder_idx % 4
            values, errs = group["Qphi"], group["Qphi_err"]
            norm_val = values.max()
            mean_val = values.mean()
            axes[axes_row, axes_col].plot(
                group["azimuth(deg)"],
                (values / mean_val - 1) * 100,
                # shadedata = errs / mean_val * 100,
                c=color_map[reg_name],
            )

    axes[1, 0].text(
        0.03,
        0.95,
        "Inner",
        c="0.3",
        fontsize=9,
        transform="axes",
        ha="left",
        va="top",
    )
    axes[0, 0].text(
        0.03,
        0.95,
        "Outer",
        c="0.3",
        fontsize=9,
        transform="axes",
        ha="left",
        va="top",
    )

    for ax in axes:
        # baseline
        ax.axhline(0, c="0.3", lw=1, zorder=0)
        # disk markers
        norm_pa = np.mod(target_info.pos_angle - 90, 360)
        ax.axvline(norm_pa, c="0.5", lw=0.5, zorder=0)
        ax.axvline(norm_pa - 180, c="0.5", lw=0.5, zorder=0)

    for i, label in enumerate(labels):
        row = 0 if i < 4 else 2
        col = i % 4
        axes[row, col].format(title=label)

    # axes[-1].legend(ncols=1, fontsize=8, order="F")

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
