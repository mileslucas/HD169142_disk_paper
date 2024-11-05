import paths
from astropy.io import fits
from utils_plot_mosaic import plot_mosaic
import proplot as pro

pro.ion()

stokes_path = paths.data / "20230707_HD169142_vampires_stokes_cube_opt_Qphi.fits"
stokes_cube, header = fits.getdata(stokes_path, header=True)

fig = plot_mosaic(stokes_cube)

axes = fig.axes
axes[-1].annotate("Inner ring", (0.1, 0.25), color="w", ha="left")
axes[-1].annotate("Outer ring", (0.2, -0.4), color="w", ha="left")

# pro.show(block=True)
fig.savefig(
    paths.figures / "20230707_HD169142_Qphi_mosaic_proposal.pdf",
    dpi=300,
)

