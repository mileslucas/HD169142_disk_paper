import proplot as pro
from utils_indexing import frame_radii
import numpy as np
from target_info import TargetInfo
import paths
from matplotlib import patches
from matplotlib.ticker import MaxNLocator
from astropy.io import fits

pro.rc["legend.fontsize"] = 7
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["cmap"] = "bone"
pro.rc["image.origin"] = "lower"
pro.rc["cycle"] = "ggplot"
pro.rc["axes.grid"] = False

stokes_cube_20230707 = fits.getdata(
    paths.data / "20230707_HD169142_vampires_stokes_cube_opt_Qphi.fits"
)
stokes_cube_20240729 = fits.getdata(
    paths.data / "20240729_HD169142_vampires_stokes_cube_opt_Qphi.fits"
)

Qphi_sum_20230707 = np.nansum(stokes_cube_20230707[:, 4], axis=0)
Qphi_sum_20240729 = np.nansum(stokes_cube_20240729[:, 4], axis=0)


plate_scale = 5.9  # mas / px
bar_width_au = 20
bar_width_arc = bar_width_au * TargetInfo.plx  # "

rs = frame_radii(stokes_cube_20230707) * plate_scale / 1e-3

side_length = stokes_cube_20230707.shape[-1] * plate_scale * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)


## plotting

fig, axes = pro.subplots(nrows=2, ncols=2, width="3.5in", space=0.25)


vmax = max(
    np.nanpercentile(Qphi_sum_20230707, 99.9), np.nanpercentile(Qphi_sum_20240729, 99.9)
)

im = axes[0, 0].imshow(Qphi_sum_20230707, extent=ext, vmin=0, vmax=vmax)
axes[0, 0].text(0.03, 0.92, r"Mean", transform="axes", c="white", fontsize=9)

im = axes[0, 1].imshow(Qphi_sum_20240729, extent=ext, vmin=0, vmax=vmax)


Qphi_sum_r2_20230707 = Qphi_sum_20230707 * rs**2
Qphi_sum_r2_20240729 = Qphi_sum_20240729 * rs**2

vmax = max(
    np.nanpercentile(Qphi_sum_r2_20230707, 98),
    np.nanpercentile(Qphi_sum_r2_20240729, 98),
)

im = axes[1, 0].imshow(Qphi_sum_r2_20230707, extent=ext, vmin=0, vmax=vmax)
im = axes[1, 1].imshow(Qphi_sum_r2_20240729, extent=ext, vmin=0, vmax=vmax)


axes[1, 0].text(
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
axes[0, 0].add_patch(rect)
axes[0, 0].text(
    0.55 - bar_width_arc / 2,
    -0.45,
    f'{bar_width_arc:.02f}"',
    c="white",
    ha="center",
    fontsize=8,
)
axes[0, 0].text(
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
axes[0, 1].plot((-0.53, delta[0] + -0.53), (-0.53, delta[1] + -0.53), color="w", lw=1)
axes[0, 1].text(
    delta[0] - 0.53,
    -0.53 + delta[1],
    "N",
    color="w",
    fontsize=7,
    ha="center",
    va="bottom",
)
delta = np.array((arrow_length, 0))
axes[0, 1].plot((-0.53, delta[0] + -0.53), (-0.53, delta[1] + -0.53), color="w", lw=1)
axes[0, 1].text(
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
    toplabels=("2023/07/07", "2024/07/29"),
)
axes[:, 1].format(ytickloc="none")
axes[0, :].format(xtickloc="none")

fig.savefig(
    paths.figures / "HD169142_Qphi_comparison.pdf",
    dpi=300,
    bbox_inches="tight",
)
