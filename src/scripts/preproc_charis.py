from utils_stokes import remove_unres_pol
import logging_config
import logging
import paths
import tqdm.auto as tqdm
from astropy.io import fits
from utils_indexing import frame_radii
from astropy.stats import biweight_location
import numpy as np

logger = logging.getLogger(__file__)

def measure_unres_pol_coeffs(stokes_cube):
    radii = frame_radii(stokes_cube)
    # two annuli
    # inside inner cavity
    ann_mask = radii <= 5

    # Q signal
    Qstar = np.nansum(stokes_cube[2] * ann_mask)
    IQstar = np.nansum(stokes_cube[0] * ann_mask)
    pQ = Qstar / IQstar
    # U signal
    Ustar = np.nansum(stokes_cube[3] * ann_mask)
    IUstar = np.nansum(stokes_cube[1] * ann_mask)
    pU = Ustar / IUstar
    return pQ, pU

if __name__ == "__main__":
    input_file = paths.data / "20230604_CHARIS_JHK" / "20230604_CHARIS_JHK_HD169142_stokes_cube.fits"

    stokes_cube_full, hdr = fits.getdata(input_file, header=True)

    for wl_idx in range(stokes_cube_full.shape[1]):
        pQ, pU = measure_unres_pol_coeffs(stokes_cube_full[:, wl_idx])
        remove_unres_pol(stokes_cube_full[:, wl_idx], pQ, pU)

    output_file = input_file.with_name(input_file.name.replace("_cube", "_cube_opt"))
    logger.info(f"Saving optimized stokes data to {output_file}")
    fits.writeto(output_file, stokes_cube_full, header=hdr, overwrite=True)

    stokes_cube_coll = biweight_location(stokes_cube_full, c=5, axis=1)
    pQ, pU = measure_unres_pol_coeffs(stokes_cube_coll)
    remove_unres_pol(stokes_cube_coll, pQ, pU)
    output_file = input_file.with_name(input_file.name.replace("_cube", "_collapsed"))
    logger.info(f"Saving collapsed stokes data to {output_file}")
    fits.writeto(output_file, stokes_cube_coll, header=hdr, overwrite=True)
