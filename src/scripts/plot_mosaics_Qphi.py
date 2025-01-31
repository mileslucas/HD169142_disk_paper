import paths
from astropy.io import fits
from utils_plot_mosaic import plot_mosaic, plot_rdi_mosaic


iwas = {
    "20230707": 105,
    "20240727": 59,
    "20240728": 59,
    "20240729": 59
}

for date in ("20230707", "20240729"):
    stokes_path = paths.data / date / "optimized" / f"{date}_HD169142_vampires_stokes_cube_optimized.fits"
    stokes_cube, header = fits.getdata(stokes_path, header=True)

    fig = plot_mosaic(stokes_cube, iwa=iwas[date], idx=4)

    fig.savefig(
        paths.figures / f"{date}_HD169142_Qphi_mosaic.pdf",
        dpi=300,
    )

# ### FLUX plots
# fig, axes = pro.subplots(nrows=2, width="3.5in", height="3.25in", sharey=0, space=0)

# cycle = pro.Colormap("fire")(np.linspace(0.4, 0.9, 4))
# pxscale = 5.9 / 1e3
# fwhm = 4
# radii = np.arange(0.105 / pxscale - fwhm / 2, 1.4 / pxscale, fwhm)
# for i in range(4):
#     Qphi_prof = RadialProfile(Qphi_frames[i], (center[1], center[0]), radii)
#     _mask = Qphi_prof.profile < 1e-2
#     Qphi_prof.profile[_mask] = np.nan
#     I_prof = RadialProfile(I_frames[i], (center[1], center[0]), radii)

#     common = dict(ms=2, c=cycle[i], zorder=100 + i)
#     axes[0].plot(
#         Qphi_prof.radius * pxscale, Qphi_prof.profile, m="o", label=titles[i], **common
#     )
#     axes[1].plot(
#         Qphi_prof.radius * pxscale,
#         Qphi_prof.profile / I_prof.profile * 100,
#         m="s",
#         **common,
#     )

# axes[0].legend(ncols=1)
# axes.format(
#     xlabel='radius (")',
#     grid=True,
# )
# axes[0].format(ylabel=r"$Q_\phi$ flux (Jy / arcsec$^2$)", yscale="log")
# axes[1].format(ylabel=r"$Q_\phi/I_{tot}$ flux (%)")
# fig.savefig(
#     paths.figures / "20230707_HD169142_Qphi_flux.pdf",
#     dpi=300,
# )
