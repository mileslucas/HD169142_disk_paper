import paths
from astropy.io import fits
from utils_plot_mosaic import plot_mosaic
import proplot as pro
from matplotlib import patches
import numpy as np

pro.rc["image.origin"] = "lower"
pro.rc["cmap"] = "magma"

fig, axes = pro.subplots(nrows=4, ncols=8, wspace=(0, 0.75, 0, 0.75, 0, 0.75, 0))

for idx, date in enumerate(("20230707", "20240727", "20240728", "20240729")):
    stokes_path = paths.data / f"{date}_HD169142_vampires_stokes_cube.fits"
    stokes_spcube, header = fits.getdata(stokes_path, header=True)

    side_length = stokes_spcube.shape[-1] * header["PXSCALE"] * 1e-3 / 2
    ext = (side_length, -side_length, -side_length, side_length)

    for wlidx, stokes_cube in enumerate(stokes_spcube):
        minmin = min(np.nanmin(stokes_cube[2]), np.nanmin(stokes_cube[3]))
        maxmax = max(np.nanmax(stokes_cube[2]), np.nanmax(stokes_cube[3]))

        norm = pro.DivergingNorm(vmin=minmin, vmax=maxmax)

        axes[idx, wlidx * 2].imshow(
            stokes_cube[2], cmap="icefire", norm=norm, extent=ext
        )
        axes[idx, wlidx * 2 + 1].imshow(
            stokes_cube[3], cmap="icefire", norm=norm, extent=ext
        )

        axes[idx, wlidx * 2].text(
            0.03,
            0.97,
            "Q",
            transform="axes",
            c="white",
            ha="left",
            va="top",
            fontsize=11,
        )
        axes[idx, wlidx * 2 + 1].text(
            0.03,
            0.97,
            "U",
            transform="axes",
            c="white",
            ha="left",
            va="top",
            fontsize=11,
        )

        arrow_length = 0.1
        delta = np.array((0, arrow_length))
        axes[idx, wlidx * 2 + 1].plot(
            (-0.53, delta[0] + -0.53), (-0.53, delta[1] + -0.53), color="w", lw=1
        )
        axes[idx, wlidx * 2 + 1].text(
            delta[0] - 0.53,
            -0.53 + delta[1],
            "N",
            color="w",
            fontsize=7,
            ha="center",
            va="bottom",
        )
        delta = np.array((arrow_length, 0))
        axes[idx, wlidx * 2 + 1].plot(
            (-0.53, delta[0] + -0.53), (-0.53, delta[1] + -0.53), color="w", lw=1
        )
        axes[idx, wlidx * 2 + 1].text(
            delta[0] - 0.525,
            -0.535 + delta[1],
            "E",
            color="w",
            fontsize=7,
            ha="right",
            va="center",
        )

# coronagraph mask
for ax in axes[0, :]:
    ax.scatter(
        0,
        0,
        color="white",
        alpha=0.8,
        marker="+",
        ms=20,
        lw=0.5,
        zorder=999,
    )
    circ = patches.Circle([0, 0], 105e-3, ec="white", fc="k", lw=1)
    ax.add_patch(circ)
for ax in axes[1:, :]:
    ax.scatter(
        0,
        0,
        color="white",
        alpha=0.8,
        marker="+",
        ms=20,
        lw=0.5,
        zorder=999,
    )
    circ = patches.Circle([0, 0], 54e-3, ec="white", fc="k", lw=1)
    ax.add_patch(circ)

axes.format(
    xlim=(0.6, -0.6),
    ylim=(-0.6, 0.6),
    grid=False,
    ytickloc="none",
    xtickloc="none",
)

pro.show(block=True)
