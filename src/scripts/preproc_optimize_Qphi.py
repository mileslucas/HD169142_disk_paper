from utils_stokes import optimize_Uphi_cube
import logging
import paths
import tqdm.auto as tqdm
from astropy.io import fits
from utils_indexing import frame_radii
import numpy as np

logger = logging.getLogger(__file__)


input_files = [
    paths.data / "20230707_VAMPIRES" / "20230707_HD169142_vampires_stokes_cube.fits",
    paths.data / "20240727_VAMPIRES" / "20240727_HD169142_vampires_stokes_cube.fits",
    paths.data / "20240729_VAMPIRES" / "20240729_HD169142_vampires_stokes_cube.fits",
]

for path in tqdm.tqdm(input_files):
    data, hdr = fits.getdata(path, header=True)
    data = data.copy()
    rads = frame_radii(data)
    mask = rads > 20
    masks = np.tile(mask, (4, 8, 1, 1))
    stokes_cube = optimize_Uphi_cube(data, mask=masks)
    outpath = path.parent / "optimized"
    outpath.mkdir(exist_ok=True)
    outname = outpath / path.name.replace(".fits", "_optimized.fits")
    logger.info(f"Saving optimized stokes data to {outname}")
    fits.writeto(outname, stokes_cube, header=hdr, overwrite=True)
