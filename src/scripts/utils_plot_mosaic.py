from matplotlib import patches
import proplot as pro
import numpy as np
from utils_indexing import frame_radii
from target_info import TargetInfo
from matplotlib.ticker import MaxNLocator

pro.rc["legend.fontsize"] = 7
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["cmap"] = "bone"
pro.rc["image.origin"] = "lower"
pro.rc["cycle"] = "ggplot"
pro.rc["axes.grid"] = False

titles = ("F610", "F670", "F720", "F760")
titles = ("610 nm", "670 nm", "720 nm", "760 nm")
plate_scale = 5.9  # mas / px


### Mosaic plots
def plot_mosaic(stokes_cube):
    # prepare some things we need
    Qphi_frames = stokes_cube[:, 4]
    Qphi_sum = np.nansum(Qphi_frames, axis=0)
    rs = frame_radii(stokes_cube) * plate_scale / 1e-3

    side_length = stokes_cube.shape[-1] * plate_scale * 1e-3 / 2
    ext = (side_length, -side_length, -side_length, side_length)

    bar_width_au = 20
    bar_width_arc = bar_width_au * TargetInfo.plx  # "

    fig, axes = pro.subplots(
        [[1, 2, 5], [3, 4, 6]],
        width="7in",
        hspace=0.25,
        wspace=[0.25, 0.75],
        spanx=False,
    )

    for Qphi, ax, title in zip(Qphi_frames, axes, titles):
        im = ax.imshow(Qphi, extent=ext, vmin=0, vmax=0.9 * np.nanmax(Qphi))
        ax.text(
            0.03,
            0.97,
            title,
            transform="axes",
            c="white",
            ha="left",
            va="top",
            fontsize=9,
        )

    vmax = np.nanpercentile(Qphi_sum, 99.9)

    im = axes[4].imshow(Qphi_sum, extent=ext, vmin=0, vmax=vmax)
    axes[4].text(0.03, 0.92, r"Mean", transform="axes", c="white", fontsize=9)

    Qphi_sum_r2 = Qphi_sum * rs**2
    vmax = np.nanpercentile(Qphi_sum_r2, 98)
    im = axes[5].imshow(Qphi_sum_r2, extent=ext, vmin=0, vmax=vmax)
    axes[5].text(
        0.03,
        0.92,
        r"Mean$\times r^2$",
        transform="axes",
        c="white",
        fontsize=9,
        bbox=dict(fc="k", alpha=0.6),
    )

    for ax in axes:
        # coronagraph mask
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
    # scale bar
    rect = patches.Rectangle([0.55, -0.485], -bar_width_arc, 8e-3, color="white")
    axes[2].add_patch(rect)
    axes[2].text(
        0.55 - bar_width_arc / 2,
        -0.45,
        f'{bar_width_arc:.02f}"',
        c="white",
        ha="center",
        fontsize=8,
    )
    axes[2].text(
        0.55 - bar_width_arc / 2,
        -0.55,
        f"{bar_width_au:.0f} au",
        c="white",
        ha="center",
        fontsize=8,
    )
    # compass rose
    arrow_length = 0.1
    delta = np.array((0, arrow_length))
    axes[1, 1].plot(
        (-0.53, delta[0] + -0.53), (-0.53, delta[1] + -0.53), color="w", lw=1
    )
    axes[1, 1].text(
        delta[0] - 0.53,
        -0.53 + delta[1],
        "N",
        color="w",
        fontsize=7,
        ha="center",
        va="bottom",
    )
    delta = np.array((arrow_length, 0))
    axes[1, 1].plot(
        (-0.53, delta[0] + -0.53), (-0.53, delta[1] + -0.53), color="w", lw=1
    )
    axes[1, 1].text(
        delta[0] - 0.525,
        -0.535 + delta[1],
        "E",
        color="w",
        fontsize=7,
        ha="right",
        va="center",
    )

    ## sup title
    axes.format(
        xlim=(0.6, -0.6),
        ylim=(-0.6, 0.6),
        grid=False,
        xlabel=r'$\Delta$RA (")',
        ylabel=r'$\Delta$DEC (")',
        ylocator=MaxNLocator(5, prune="both"),
        xlocator=MaxNLocator(5, prune="both"),
    )
    axes[:, 1].format(ytickloc="none")
    axes[0, :].format(xtickloc="none")

    return fig
