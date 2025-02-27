import numpy as np
from vampires_dpp.image_processing import derotate_frame, derotate_cube
import warnings
import tqdm
from utils_organization import folders
import paths
from astropy.io import fits
import logging_config
import logging

logger = logging.getLogger(__file__)

NFRAMES = 1700
TOTPA = 150

def create_faux_adi_image(frame, N=NFRAMES, pa_rot=TOTPA):
    pas = np.linspace(0, pa_rot, N)
    faux_cube = np.empty((N, *frame.shape), dtype=frame.dtype)
    for idx in range(N):
        faux_cube[idx] = derotate_frame(frame, -pas[idx])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med_frame = np.nanmedian(faux_cube, axis=0, keepdims=True)

    res_cube = faux_cube - med_frame
    derot_res_cube = derotate_cube(res_cube, pas)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_frame = np.nanmedian(derot_res_cube, axis=0, overwrite_input=True)

    return res_frame

def create_radial_profile_image(frame, radii):
    output = np.zeros_like(frame)
    radii_ints = np.round(radii).astype(int)
    bins = np.arange(radii_ints.min(), radii_ints.max() + 1)
    count_map = {}
    for bin in bins:
        mask = (radii_ints == bin) & np.isfinite(frame)
        data = frame[mask]
        mean = np.nanmean(data)
        count_map[bin] = mean
    for ridx in range(output.shape[0]):
        for cidx in range(output.shape[1]):
            radius = radii[ridx, cidx]
            bin = np.round(radius).astype(int)
            value = count_map[bin]
            output[ridx, cidx] = value

    return output

def create_radial_noise_image(frame, radii):
    output = np.zeros_like(frame)
    radii_ints = np.round(radii).astype(int)
    bins = np.arange(radii_ints.min(), radii_ints.max() + 1)
    count_map = {}
    for bin in bins:
        mask = (radii_ints == bin) & np.isfinite(frame)
        data = frame[mask]
        mean = np.nanstd(data)
        count_map[bin] = mean
    for ridx in range(output.shape[0]):
        for cidx in range(output.shape[1]):
            radius = radii[ridx, cidx]
            bin = np.round(radius).astype(int)
            value = count_map[bin]
            output[ridx, cidx] = value

    return output

if __name__ == "__main__":
    pbar = tqdm.tqdm(folders)
    for i, folder in enumerate(pbar):
        stokes_path = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Qphi_r2_scaled.fits"
        logger.info(f"Creating faux ADI image for {folder} with {NFRAMES} frames and {TOTPA} deg PA rotation")
        Qphi_image, header = fits.getdata(stokes_path, header=True)
        radius_map = fits.getdata(paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Qphi_radius.fits")
        faux_adi_image = create_faux_adi_image(Qphi_image / radius_map**2)

        header["hierarch CADI NFRAMES"] = NFRAMES
        header["hierarch CADI TOTPA"] = TOTPA
        filename = paths.data / folder / f"{folder}_HD169142_Qphi_cADI_sim.fits"
        logger.info(f"Saving output to {filename.name}")
        fits.writeto(filename, faux_adi_image, header=header, overwrite=True)
        filename = paths.data / folder / f"{folder}_HD169142_Qphi_cADI_sim_r2_scaled.fits"
        logger.info(f"Saving output to {filename.name}")
        fits.writeto(filename, faux_adi_image * radius_map**2, header=header, overwrite=True)