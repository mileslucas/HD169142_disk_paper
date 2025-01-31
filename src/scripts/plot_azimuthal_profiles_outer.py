import proplot as pro
import numpy as np
import paths
import tqdm
import pandas as pd

pro.rc["legend.facecolor"] = pro.rc["axes.facecolor"]
pro.rc["image.origin"] = "lower"
pro.rc["image.cmap"] = "magma"

dates = ("20230707", "20240729")
names = ["F610", "F670", "F720", "F760"]
dist = 114.5
pxscale = 5.9e-3
## Plot and save

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


## Plot and save
fig, axes = pro.subplots(
    nrows=len(names), ncols=len(dates), width="7in", height="3.5in", wspace=0.5, hspace=0, spanx=False
)


cycle = pro.Colormap("fire")(np.linspace(0.4, 0.9, 4))

for i, date in enumerate(tqdm.tqdm(dates)):
    # load data
    table = pd.read_csv(
        paths.data / date / f"{date}_HD169142_vampires_azimuthal_profiles.csv"
    )
    subtable = table.query("region == 'outer'")
    groups = subtable.groupby("filter")
    for wl_idx, filt_name in enumerate(names):
        group = groups.get_group(filt_name)

        azimuth = group["azimuth(deg)"]
        values, errs = moving_average(group["Qphi"], group["Qphi_err"], window_size=10)
        normval = values.max()
        values /= normval
        errs /= normval
        meanval = values.mean()

        axes[wl_idx, i].plot(
            azimuth,
            values,
            shadedata=errs,
            c=cycle[wl_idx],
            label=names[wl_idx],
        )

        axes[wl_idx, i].text(
            0.01,
            0.97,
            filt_name,
            transform="axes",
            va="top",
            ha="left",
            c=cycle[wl_idx],
            fontsize=8,
        )

for ax in axes:
    ax.axhline(0, c="0.3", lw=1, zorder=0)
# ## sup title
axes.format(
    ylim=(-0.25, None),
    # xlim=(0, 150),
    xlabel="Azimuth (° East of North)",
    ylabel=r"Normalized azimuthal profile $\times r^2$",
    xlocator=45,
    ylocator=0.5,
    toplabels=["/".join((date[:-4], date[4:6], date[6:])) for date in dates]
)

axes[:, -1].format(ytickloc="none")
axes[:-1, :].format(xtickloc="none")
fig.savefig(
    paths.figures / "HD169142_azimuthal_profiles_outer.pdf",
    bbox_inches="tight",
    dpi=300,
)
