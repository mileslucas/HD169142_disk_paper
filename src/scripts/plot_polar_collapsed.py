import proplot as pro
import numpy as np
import paths
from astropy.io import fits
from skimage.transform import warp_polar
from astropy.convolution import convolve, kernels
import tqdm
from astropy.visualization import simple_norm

pro.use_style("ggplot")
pro.rc["legend.facecolor"] = pro.rc["axes.facecolor"]
pro.rc["image.origin"] = "lower"
pro.rc["image.cmap"] = "magma"
pro.rc["axes.grid"] = False
pro.rc["axes.facecolor"] = "k"

folders = ("20230707_VAMPIRES", "20240729_VAMPIRES")

iwas = {
    "20230707_VAMPIRES": 105,
    "20240727_VAMPIRES": 59,
    "20240728_VAMPIRES": 59,
    "20240729_VAMPIRES": 59
}

fwhm = 4
kernel_width = fwhm / (2 * np.sqrt(2 * np.log(2)))
## Plot and save
fig, axes = pro.subplots(
    nrows=2, width="3.5in", refheight="1.5in", space=0.25, sharey=3, sharex=3
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

    plate_scale = 5.9e-3
    dist = 114.8
    ext = (0, 360, 0, polar_cube.shape[-2] * plate_scale * dist)

    # PDI images
    norm = simple_norm(polar_cube, stretch="sinh", vmin=0)
    im = axes[i].imshow(polar_cube, extent=ext, norm=norm, origin="lower", cmap="magma")
    # axes[0].colorbar(im)
    axes[i].text(
        0.03, 0.95, format_date(folder.split("_")[0]), transform="axes", c="white", ha="left", va="top", fontsize=8
    )

    axes[i].axhline(iwas[folder] / 1e3 * dist, c="w", alpha=0.4)

## sup title
axes.format(
    ylim=(0, 115),
    aspect="auto",
    xlabel="Angle E of N (°)",
    ylabel="Separation (au)",
    xlocator=90,
)

axes[:-1].format(xtickloc="none")

fig.savefig(
    paths.figures / f"{folder}_HD169142_vampires_polar_collapsed.pdf", bbox_inches="tight", dpi=300
)

####################################################################################
####################################################################################
####################################################################################


# radii = np.linspace(ext[-2], ext[-1], polar_frames.shape[-1])
# T0 = 75  # deg/yr at 21 au
# periods = np.sqrt(T0**2 * (radii / 21) ** 3)
# rates = -360 / periods
# rates[0] = 0
# dt = 1  # yr
# change = rates * dt
# int_change = np.round(change).astype(int)


# im2_roll = im2.copy()
# for i in range(im2.shape[0]):
#     im2_roll[i] = np.roll(im2[i], (-int_change[i], 0))

# im = axes[0].imshow(im1, extent=ext, vmin=0)
# im = axes[1].imshow(im2_roll, extent=ext, vmin=0)
# # axes[0].colorbar(im)
# axes[0].text(
#     0.03, 0.95, "2023/07", transform="axes", c="white", ha="left", va="top", fontsize=8
# )
# axes[1].text(
#     0.03, 0.95, "2024/07", transform="axes", c="white", ha="left", va="top", fontsize=8
# )

# axes[0].axhline(0.105 * dist, c="w", alpha=0.4)
# axes[1].axhline(0.059 * dist, c="w", alpha=0.4)

# ## sup title
# axes.format(
#     ylim=(0, 37),
#     aspect="auto",
#     xlabel="Angle E of N (°)",
#     ylabel="Separation (au)",
#     xlocator=90,
#     ylocator=10,
# )

# axes[:-1].format(xtickloc="none")

# sep = 248.6
# axes[0].axvline(sep, c="#389826", alpha=0.7)
# axes[1].axvline(sep, c="#389826", alpha=0.7)

# fig.savefig(
#     paths.figures / "vampires_polar_collapsed_rolled.pdf", bbox_inches="tight", dpi=300
# )
