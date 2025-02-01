import proplot as pro
import numpy as np
import paths
from astropy.io import fits
from skimage.transform import warp_polar
from astropy.convolution import convolve, kernels
import tqdm
from astropy.visualization import simple_norm
from target_info import target_info
from utils_ephemerides import blob_c_position, blob_d_position, keplerian_warp
from astropy import time


def label_from_folder(foldername):
    tokens = foldername.split("_")
    date = f"{tokens[0][:4]}/{tokens[0][4:6]}/{tokens[0][6:]}"
    return f"{date} {tokens[1]}"

def time_from_folder(foldername: str) -> time.Time:
    date_raw = foldername.split("_")[0]
    ymd = {
        "year": int(date_raw[:4]),
        "month": int(date_raw[4:6]),
        "day": int(date_raw[6:])
    }
    return time.Time(ymd, format="ymdhms")


if __name__ == "__main__":
    pro.rc["image.origin"] = "lower"
    pro.rc["image.cmap"] = "bone"
    pro.rc["axes.grid"] = False
    pro.rc["axes.facecolor"] = "w"
    pro.rc["font.size"] = 9
    pro.rc["title.size"] = 9

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

    ## Plot and save
    fig, axes = pro.subplots(
        nrows=8, width="3.33in", refheight="0.8in", hspace=0.5
    )

    def format_date(date):
        return f"{date[:4]}/{date[4:6]}"

    timestamps = list(map(time_from_folder, folders))

    t0 = timestamps[4]

    for i, folder in enumerate(tqdm.tqdm(folders)):
    # load data
        with fits.open(
            paths.data
            / folder
            / f"{folder}_HD169142_Qphi_polar.fits"
        ) as hdul:
            polar_cube = hdul[0].data

        rin = np.floor(15 / target_info.dist_pc / pxscales[folder]).astype(int)
        rout = np.ceil(45 / target_info.dist_pc / pxscales[folder]).astype(int)
        
        rs = np.arange(polar_cube.shape[0])

        mask = (rs >= rin) & (rs <= rout)
        ext = (0, 360, rin * pxscales[folder] * target_info.dist_pc, rout * pxscales[folder] * target_info.dist_pc)

        c_a, c_th = blob_c_position(t0)
        d_a, d_th = blob_d_position(t0)
        

        axes[i].scatter(c_th, c_a, marker="^", ms=30, c="0.1", lw=1)
        axes[i].scatter(d_th, d_a, marker="v", ms=30, c="0.1", lw=1)

        # PDI images
        image = keplerian_warp(polar_cube[mask, :], rs[mask] * target_info.dist_pc * pxscales[folder], timestamps[i], t0)
        image_mean = np.nanmedian(image, axis=1, keepdims=True)
        norm_image = image  - image_mean

        norm = pro.DivergingNorm()
        im = axes[i].imshow(norm_image, extent=ext, cmap="div", norm=norm)
        # axes[0].colorbar(im)
        labels = label_from_folder(folder).split()
        axes[i].text(
            0.03, 0.95, labels[0], transform="axes", c="0.1 ", ha="left", va="top", fontsize=8, fontweight="bold"
        )
        axes[i].text(
            0.99, 0.95, labels[1], transform="axes", c="0.1 ", ha="right", va="top", fontsize=8, fontweight="bold"
        )

        # axes[i].axhline(iwas[folder] / 1e3 * dist, c="w", alpha=0.4)

    



    ## sup title
    axes.format(
        aspect="auto",
        xlabel="Angle E of N (°)",
        ylabel="Separation (au)",
        xlocator=90,
    )

    # axes[:-1].format(xtickloc="none")

    fig.savefig(
        paths.figures / "HD169142_polar_collapsed_inner_rolled_subbed.pdf", bbox_inches="tight", dpi=300
    )
