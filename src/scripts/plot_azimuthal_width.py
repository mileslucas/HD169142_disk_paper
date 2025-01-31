import proplot as pro
import numpy as np
import paths
from astropy.io import fits
from skimage.transform import warp_polar
from astropy.convolution import convolve, kernels
from astropy.visualization import simple_norm
from astropy.modeling import models, fitting
from matplotlib import ticker
import tqdm
import pandas as pd


dates = ("20230707", "20240729")
filters = ("F610", "F670", "F720", "F760")

## Plot and save
fig, axes = pro.subplots(
    nrows=4,
    ncols=2,
    width="7in",
    height="3in", 
    wspace=0.5,
    hspace=0,
    sharey=4,
    spanx=False,
)
cycle = pro.Colormap("fire")(np.linspace(0.4, 0.9, 4))

def moving_average(arr, err, window_size):
    # pad edges of array with wrapping
    arr_pad = np.pad(arr, window_size, mode="wrap")
    var_pad = np.pad(err**2, window_size, mode="wrap")

    window = np.ones(window_size)
    norm_window = window / window.sum()
    vals = np.convolve(arr_pad, norm_window, "valid")[
        window_size // 2 : -window_size // 2 - 1
    ]
    vars = np.convolve(var_pad, norm_window, "valid")[
        window_size // 2 : -window_size // 2 - 1
    ]
    return vals, np.sqrt(vars / window_size)

for i, date in enumerate(tqdm.tqdm(dates)):
    # load data
    table = pd.read_csv(
        paths.data / date / f"{date}_HD169142_vampires_azimuthal_profiles.csv"
    )
    subtable = table.query("region == 'inner'")
    groups = subtable.groupby("filter")

    # load data
    with fits.open(
        paths.data
        / date
        / f"{date}_HD169142_vampires_Qphi_polar.fits"
    ) as hdul:
        polar_cube = hdul[0].data


    plate_scale = 5.9e-3
    dist = 114.8

    min_idx = int(0.11 / plate_scale)
    max_idx = int(35 / dist / plate_scale)
    polar_cut = polar_cube[:, min_idx:max_idx]

    for wl_idx in range(polar_cut.shape[0]):
        group = groups.get_group(filters[wl_idx])

        azimuth = group["azimuth(deg)"]
        values, errs = moving_average(group["Qphi"], group["Qphi_err"], window_size=10)
        normval = values.max()
        values /= normval
        meanval = values.mean()


        ring_fwhm = []
        ring_fwhm_errs = []
        for az_idx in range(polar_cut.shape[-1]):
            _slice = polar_cut[wl_idx, :, az_idx]
            mu = np.argmax(_slice)
            model = models.Gaussian1D(amplitude=_slice[mu], mean=mu, stddev=2)
            fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)
            res = fitter(model, np.indices(_slice.shape)[0], _slice)
            ring_fwhm.append(res.fwhm)
            cov_matrix = fitter.fit_info['param_cov']
            uncertainties = np.sqrt(np.diag(cov_matrix))
            ring_fwhm_errs.append(uncertainties[2] * 2 * np.sqrt(2 * np.log(2)))

        # inner ring brightness line
        ax2 = axes[wl_idx, i].twinx()
        ax2.plot(
            azimuth,
            values,
            lw=1,
            c="0.5",
            zorder=0
        )
        # hide the new spine and restore original spine w/o ticks
        ax2.format(yspineloc="none")
        axes[wl_idx, i].spines["right"].set(visible=True)

        # peak radius line
        axes[wl_idx, i].plot(
            np.linspace(0, 360, polar_cube.shape[-1]),
            np.array(ring_fwhm) * plate_scale * dist,
            shadedata=np.array(ring_fwhm_errs) * plate_scale * dist,
            c = cycle[wl_idx],
            zorder=10
        )


        # filter label
        axes[wl_idx, i].text(
            0.01,
            0.95,
            filters[wl_idx],
            transform="axes",
            va="top",
            ha="left",
            c=cycle[wl_idx],
            fontsize=8,
        )

axes[:, 1:].format(ytickloc="none")
axes[:-1, :].format(xtickloc="none")
## sup title
axes.format(
    xlabel="Angle E of N (°)",
    ylabel="Width of inner ring (au)",
    xlocator=range(0, 360, 45),
    toplabels=["/".join((date[:-4], date[4:6], date[6:])) for date in dates],
    titlesize=9,
)


fig.savefig(paths.figures / "HD169142_vampires_azimuthal_width.pdf", bbox_inches="tight", dpi=300)
