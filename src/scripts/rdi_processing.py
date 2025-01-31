import paths
from astropy.io import fits
import numpy as np
from utils_rdi import rdi

## 2023/07/07
sci_cube, hdr = fits.getdata(
    paths.data / "20230707" / "20230707_HD169142_vampires_adi_cube.fits", header=True
)
pas = fits.getdata(
    paths.data / "20230707" / "20230707_HD169142_vampires_adi_angles.fits"
)
ref_cube = fits.getdata(
    paths.data / "20230707" / "20230707_HD169141_vampires_adi_cube.fits"
)

# minimum sep in pixels for optimization zone
# typically IWA + a few pixels
rmin = 105 // 5.9
# maximum sep in pixels for optimization zone
rmax = 100
# image half-width for cropping
half_width = 200
imcube0 = rdi(sci_cube, ref_cube, pas, rmin=rmin, rmax=rmax, half_width=half_width)
outdir = paths.data / "20230707" / "rdi_residuals"
outdir.mkdir(exist_ok=True)
fits.writeto(
    outdir / "20230707_HD169142_vampires_rdi_cube.fits",
    imcube0,
    header=hdr,
    overwrite=True,
)
fits.writeto(
    outdir / "20230707_HD169142_vampires_rdi_frame.fits",
    np.nanmean(imcube0, axis=0),
    header=hdr,
    overwrite=True,
)


## 2024/07/27
sci_cube, hdr = fits.getdata(
    paths.data / "20240727" / "20240727_HD169142_vampires_adi_cube.fits", header=True
)
pas = fits.getdata(
    paths.data / "20240727" / "20240727_HD169142_vampires_adi_angles.fits"
)
ref_cube = fits.getdata(
    paths.data / "20240727" / "20240727_HD168453_vampires_adi_cube.fits"
)

# minimum sep in pixels for optimization zone
# typically IWA + a few pixels
rmin = 69 // 5.9 + 2
imcube0 = rdi(sci_cube, ref_cube, pas, rmin=rmin, rmax=rmax, half_width=half_width)
outdir = paths.data / "20240727" / "rdi_residuals"
outdir.mkdir(exist_ok=True)
fits.writeto(
    outdir / "20240727_HD169142_vampires_rdi_cube.fits",
    imcube0,
    header=hdr,
    overwrite=True,
)
fits.writeto(
    outdir / "20240727_HD169142_vampires_rdi_frame.fits",
    np.nanmean(imcube0, axis=0),
    header=hdr,
    overwrite=True,
)


## 2024/07/28
sci_cube, hdr = fits.getdata(
    paths.data / "20240728" / "20240728_HD169142_vampires_adi_cube.fits", header=True
)
pas = fits.getdata(
    paths.data / "20240728" / "20240728_HD169142_vampires_adi_angles.fits"
)
ref_cube = fits.getdata(
    paths.data / "20240728" / "20240728_HD317501_vampires_adi_cube.fits"
)

# minimum sep in pixels for optimization zone
# typically IWA + a few pixels
imcube0 = rdi(sci_cube, ref_cube, pas, rmin=rmin, rmax=rmax, half_width=half_width)
outdir = paths.data / "20240728" / "rdi_residuals"
outdir.mkdir(exist_ok=True)
fits.writeto(
    outdir / "20240728_HD169142_vampires_rdi_cube.fits",
    imcube0,
    header=hdr,
    overwrite=True,
)
fits.writeto(
    outdir / "20240728_HD169142_vampires_rdi_frame.fits",
    np.nanmean(imcube0, axis=0),
    header=hdr,
    overwrite=True,
)

## 2024/07/29
sci_cube, hdr = fits.getdata(
    paths.data / "20240729" / "20240729_HD169142_vampires_adi_cube.fits", header=True
)
pas = fits.getdata(
    paths.data / "20240729" / "20240729_HD169142_vampires_adi_angles.fits"
)
ref_cube = fits.getdata(
    paths.data / "20240729" / "20240729_HD317501_vampires_adi_cube.fits"
)

# minimum sep in pixels for optimization zone
# typically IWA + a few pixels
imcube0 = rdi(sci_cube, ref_cube, pas, rmin=rmin, rmax=rmax, half_width=half_width)
outdir = paths.data / "20240729" / "rdi_residuals"
outdir.mkdir(exist_ok=True)
fits.writeto(
    outdir / "20240729_HD169142_vampires_rdi_cube.fits",
    imcube0,
    header=hdr,
    overwrite=True,
)
fits.writeto(
    outdir / "20240729_HD169142_vampires_rdi_frame.fits",
    np.nanmean(imcube0, axis=0),
    header=hdr,
    overwrite=True,
)
