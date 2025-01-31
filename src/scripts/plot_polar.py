import proplot as pro
import numpy as np
import paths
from astropy.io import fits
from skimage.transform import warp_polar
from astropy.convolution import convolve, kernels
from astropy.visualization import simple_norm
import tqdm

pro.rc["image.origin"] = "lower"
pro.rc["image.cmap"] = "magma"
pro.rc["axes.grid"] = False

dates = ("20230707", "20240729")

## Plot and save
fig, axes = pro.subplots(
    nrows=len(dates),
    ncols=4,
    refwidth="2in",
    refheight="1in",
    wspace=0.5,
    hspace=0.5,
)

for i, date in enumerate(tqdm.tqdm(dates)):
    # load data
    with fits.open(
        paths.data
        / date
        / f"{date}_HD169142_vampires_Qphi_polar.fits"
    ) as hdul:
        polar_cube = hdul[0].data
    plate_scale = 5.9e-3
    dist = 114.8
    ext = (0, 360, 0, polar_cube.shape[-2] * plate_scale * dist)

    # PDI images
    for j in range(polar_cube.shape[0]):
        norm=simple_norm(polar_cube[j], vmin=0, stretch="sinh")
        im = axes[i, j].imshow(polar_cube[j], extent=ext, norm=norm, cmap="magma")
    # axes[0].colorbar(im)
    axes[i, 0].text(
        0.03,
        0.95,
        "/".join((date[:-4], date[4:6], date[6:])),
        transform="axes",
        c="white",
        ha="left",
        va="top",
        fontsize=8,
    )

labels = ("610 nm", "670 nm", "720 nm", "760 nm")
for i in range(4):
    axes[0, i].axhline(0.105 * dist, c="w", alpha=0.4)
    axes[1, i].axhline(0.059 * dist, c="w", alpha=0.4)
    axes[0, i].format(title=labels[i])

axes[:, 1:].format(ytickloc="none")
axes[:-1, :].format(xtickloc="none")
## sup title
axes.format(
    aspect="auto",
    ylim=(None, 100),
    xlabel="Angle E of N (°)",
    ylabel="Separation (au)",
    xlocator=45,
    # ylocator=10,
    titlesize=9,
)


fig.savefig(paths.figures / "HD169142_vampires_polar.pdf", bbox_inches="tight", dpi=300)


# for i in range(4):
#     axes[0, i].axvline(-6, c="#CB3C33", alpha=0.8)
#     axes[1, i].axvline(-15, c="#CB3C33", alpha=0.8)


# fig.savefig(
#     paths.figures / "vampires_polar_annotated.pdf", bbox_inches="tight", dpi=300
# )
