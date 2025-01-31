import proplot as pro
import numpy as np
import paths
import tqdm
import pandas as pd
import fitting_radial_profile as fitting
from astropy.io import fits
from extract_radial_profile import get_radial_profile
from utils_indexing import frame_radii

pro.rc["legend.facecolor"] = pro.rc["axes.facecolor"]
pro.rc["image.origin"] = "lower"
pro.rc["image.cmap"] = "magma"

dates = ("20230707", "20240729")
names = ["F610", "F670", "F720", "F760"]
dist = 114.8
pxscale = 5.9e-3
iwas = {"20230707": 105, "20240727": 59, "20240728": 59, "20240729": 59}

## Plot and save
fig, axes = pro.subplots(
    nrows=2,
    ncols=1,
    width="3.5in",
    refheight="2in",
    space=0.5,
    hratios=(0.7, 0.3),
    span=False,
)


alma_data, alma_hdr = fits.getdata(paths.data / "HD169142.selfcal.concat.GPU-UVMEM.centered_mJyBeam.fits", header=True)
alma_pxscale = np.abs(alma_hdr["CDELT1"]) * 3.6e3 # arcsec / px
alma_rs = frame_radii(alma_data)
alma_noise = np.ones_like(alma_data)
alma_profile = get_radial_profile(alma_data, alma_noise, alma_rs)

norm_profile = alma_profile["profile"] / np.nanmax(alma_profile["profile"])

cycle = pro.Colormap("fire")(np.linspace(0.4, 0.9, 4))
colors = ["r", "b"]
for date_idx, date in enumerate(tqdm.tqdm(dates)):
    # load data
    table = pd.read_csv(
        paths.data / date / f"{date}_HD169142_vampires_radial_profiles.csv"
    )

    groups = table.groupby("filter")
    values = []
    errs = []
    for wl_idx, (filt_name, group) in enumerate(groups):
        radius = group["radius(au)"].values
        values.append(group["Qphi"].values)
        errs.append(group["Qphi_err"].values)

    mean_profile = np.nanmean(values, axis=0)
    mean_std = np.nanstd(values, axis=0)
    mean_err = np.sqrt(np.nanmean(np.power(errs, 2)) + mean_std**2)
    filename = paths.data / date / f"{date}_HD169142_vampires_radial_profile_posteriors.npz"
    posterior_dict = np.load(filename)
    med_post = np.median(posterior_dict["samples"], axis=0)

    model = fitting.model_two_double_powerlaw_rings(radius, med_post)
    mean_profile /= np.nanmax(mean_profile)
    model /= np.max(model)
    
    axes[0].scatter(
        radius,
        mean_profile,
        c=colors[date_idx],
        label= "/".join((date[:-4], date[4:6], date[6:])),
        marker=".",
        markersize=10,
    )
    axes[0].plot(
        radius,
        model,
        c=colors[date_idx],
        lw=1
    )

    axes[1].scatter(
        radius,
        mean_profile - model,
        c=colors[date_idx],
        marker=".",
        markersize=10,
    )

    axes[0].axvline(iwas[date] * 1e-3 * dist, c=colors[date_idx], ls="--", lw=1)
    axes[1].axvline(iwas[date] * 1e-3 * dist, c=colors[date_idx], ls="--", lw=1)

axes[0].plot(alma_profile["radius"] * dist * alma_pxscale, norm_profile, c="0.5", zorder=0, lw=1, label="ALMA")

axes[0].legend(ncols=1, fontsize=8, order="F")

axes[0].axhline(0, c="0.1", lw=1, zorder=0)
axes[1].axhline(0, c="0.1", lw=1, zorder=0)

# ## sup title
axes[0].format(
    ylabel=r"Normalized radial profile $\times r^2$",
)
axes[1].format(
    ylabel=r"Residual",
    ylim=(-0.12, 0.12)
)
axes.format(
    xlim=(0, 125),
    xlabel="Separation (au)",
)

axes[:-1].format(xtickloc="none")
fig.savefig(
    paths.figures / "HD169142_radial_profiles_with_model.pdf",
    bbox_inches="tight",
    dpi=300,
)
