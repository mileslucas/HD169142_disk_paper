import proplot as pro
import numpy as np
import paths
from astropy.io import fits
from astropy.convolution import convolve, kernels
import tqdm
from astropy.stats import biweight_location, biweight_scale
import polarTransform as pt
from scipy.signal import savgol_filter
import pandas as pd

pro.rc["legend.facecolor"] = pro.rc["axes.facecolor"]
pro.rc["image.origin"] = "lower"
pro.rc["image.cmap"] = "magma"

dates = ("20230707", "20240729")
names = ["F610", "F670", "F720", "F760"]
dist = 114.5
pxscale = 5.9e-3
iwas = {"20230707": 105, "20240727": 59, "20240728": 59, "20240729": 59}

## Plot and save
fig, axes = pro.subplots(nrows=len(names), ncols=len(dates), width="7in", height="3.75in", hspace=0, wspace=0.5, spanx=False)


cycle = pro.Colormap("fire")(np.linspace(0.4, 0.9, 4))

for i, date in enumerate(tqdm.tqdm(dates)):
    # load data
    table = pd.read_csv(paths.data / date / f"{date}_HD169142_vampires_radial_profiles.csv")
    groups = table.groupby("filter")
    for wl_idx, filt_name in enumerate(names):
        group = groups.get_group(filt_name)
        normval = group["Qphi"].max()


        radius = group["radius(au)"]
        axes[wl_idx, i].plot(
            radius,
            group["Qphi"] / normval,
            shadedata=group["Qphi_err"] / normval,
            # shadedata=err / normval,
            c=cycle[wl_idx],
            label=names[wl_idx],
        )
        rmax = radius.iloc[group["Qphi"].argmax()]
        axes[wl_idx, i].plot(
            [rmax, rmax],
            [0, 1],
            c=cycle[wl_idx],
            lw=1,
            alpha=0.8
            
        )
        # estimate satellite spot
        satspot = 15.9 * np.rad2deg(float(filt_name[1:]) / 7.92e9) * 3.6e3 * dist
        yval = group["Qphi"].loc[group["radius(au)"] > satspot].values[0] / normval
        axes[wl_idx, i].scatter(
            [satspot], [yval + 0.3],
            marker="^",
            ms=10,
            c=cycle[wl_idx]
        )

        # estimate control ring
        control_ring = 22.5 * np.rad2deg(float(filt_name[1:]) / 7.92e9) * 3.6e3 * dist
        yval = group["Qphi"].loc[group["radius(au)"] > control_ring].values[0] / normval
        axes[wl_idx, i].scatter(
            [control_ring], [yval + 0.3],
            marker="d",
            ms=11,
            c=cycle[wl_idx]
        )

        yval = group["Qphi"].loc[group["radius(au)"] > 145].values[0] / normval
        axes[wl_idx, i].text(
            145, yval + 0.4, filt_name,
            va="top",
            ha="right",
            c=cycle[wl_idx],
            fontsize=8
        )

        axes[wl_idx, i].axvline(iwas[date] * 1e-3 * dist, c="0.3", ls="--", lw=1)

for ax in axes:
    ax.axhline(0, c="0.3", lw=1, zorder=0)

# ## sup title
axes.format(
    ylim=(-0.25, None),
    xlim=(0, 150),
    xlabel="Separation (au)",
    ylabel=r"Normalized radial profile $\times r^2$",
    toplabels=[ "/".join((date[:-4], date[4:6], date[6:])) for date in dates],
    # yscale="log",
    # yformatter="log",
)

axes[:-1, :].format(xtickloc="none")
axes[:, -1].format(ytickloc="none")
fig.savefig(
    paths.figures / "HD169142_radial_profiles.pdf", bbox_inches="tight", dpi=300
)
