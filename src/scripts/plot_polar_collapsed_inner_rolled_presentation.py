import proplot as pro
import numpy as np
import paths
from astropy.io import fits
import tqdm
from astropy.visualization import simple_norm
from astropy import time

from target_info import target_info
from utils_ephemerides import keplerian_warp


def time_from_folder(foldername: str) -> time.Time:
    date_raw = foldername.split("_")[0]
    ymd = {
        "year": int(date_raw[:4]),
        "month": int(date_raw[4:6]),
        "day": int(date_raw[6:])
    }
    return time.Time(ymd, format="ymdhms")
def label_from_folder(foldername):
    tokens = foldername.split("_")
    date = f"{tokens[0][:4]}/{tokens[0][4:6]}/{tokens[0][6:]}"
    return f"{date} {tokens[1]}"


if __name__ == "__main__":
    pro.rc["image.origin"] = "lower"
    pro.rc["image.cmap"] = "bone"
    pro.rc["axes.grid"] = False
    pro.rc["axes.facecolor"] = "k"
    pro.rc["font.size"] = 8
    pro.rc["title.size"] = 9
    pro.rc["figure.dpi"] = 300

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
    iwas = {
        "20230707_VAMPIRES": 105,
        "20240727_VAMPIRES": 59,
        "20240728_VAMPIRES": 59,
        "20240729_VAMPIRES": 59
    }

    pxscales = {
        "20120726_NACO": 27e-3,
        "20140425_GPI": 14.14e-3,
        "20150503_IRDIS": 12.25e-3,
        "20150710_ZIMPOL": 3.6e-3,
        "20180715_ZIMPOL": 3.6e-3,
        "20230707_VAMPIRES": 5.9e-3,
        "20210906_IRDIS": 12.25e-3,
        "20240729_VAMPIRES": 5.9e-3,
    }
    timestamps = list(map(time_from_folder, folders))
    ## Plot and save
    height = 3.31314
    width = 2.3 * height
    fig, axes = pro.subplots(
        ncols=8, height=f"{height}in", width=f"{width}in", wspace=0.5
    )

    def format_date(date):
        return f"{date[:4]}/{date[4:6]}"

    for i, folder in enumerate(tqdm.tqdm(folders)):
    # load data
        with fits.open(
            paths.data
            / folder
            / f"{folder}_HD169142_Qphi_polar.fits"
        ) as hdul:
            polar_cube = hdul[0].data

        rin = np.floor(15 / target_info.dist_pc / pxscales[folder]).astype(int)
        rout = np.ceil(35 / target_info.dist_pc / pxscales[folder]).astype(int)

        rs = np.arange(polar_cube.shape[0])

        mask = (rs >= rin) & (rs <= rout)
        ext = (rin * pxscales[folder] * target_info.dist_pc, rout * pxscales[folder] * target_info.dist_pc, 360, 0)

        rs_au = rs[mask] * target_info.dist_pc * pxscales[folder]
        polar_cube_rolled = keplerian_warp(polar_cube[mask, :], rs_au, timestamps[i], timestamps[4])


        # PDI images
        data = np.flipud(polar_cube_rolled.T)
        norm = simple_norm(data, vmin=0, stretch="sinh", sinh_a=0.5)
        im = axes[i].imshow(data, extent=ext, norm=norm, vmin=norm.vmin, vmax=norm.vmax, cmap=pro.rc["cmap"])
        # axes[0].colorbar(im)
        labels = label_from_folder(folder).split()
        axes[i].text(
            0.95, 0.99, labels[0], transform="axes", c="white", ha="right", va="top", fontsize=8, fontweight="bold", rotation=-90
        )
        axes[i].text(
            0.95, 0.01, labels[1], transform="axes", c="white", ha="right", va="bottom", fontsize=8, fontweight="bold", rotation=-90
        )

        # axes[i].axhline(iwas[folder] / 1e3 * dist, c="w", alpha=0.4)

    # for ax in axes:
    #     norm_pa = np.mod(target_info.pos_angle - 90, 360)
    #     ax.axvline(norm_pa, lw=1, c="0.8")
    #     ax.axvline(norm_pa - 180, lw=1, c="0.8")


    ## sup title
    axes.format(
        aspect="auto",
        ylabel="Angle E of N (°)",
        xlabel="Separation (au)",
        ylocator=90,
    )
    axes[1:].format(ytickloc="none")


    fig.savefig(
        paths.figures / "HD169142_polar_collapsed_inner_rolled_presentation.pdf", bbox_inches="tight"
    )
