import numpy as np
import paths
from astropy.io import fits
import tqdm
from astropy.convolution import convolve, kernels
import polarTransform as pt
from utils_indexing import frame_radii
from target_info import target_info
from utils_ephemerides import _DEG_PER_PIXEL
from utils_organization import pxscales

if __name__ == "__main__":

    folders = [
        "20120726_NACO_H",
        "20140425_GPI_J",
        "20150503_IRDIS_J",
        "20150710_ZIMPOL_VBB",
        "20170918_ALMA_1.3mm",
        "20180715_ZIMPOL_VBB",
        "20210906_IRDIS_Ks",
        "20230604_CHARIS_JHK",
        "20230707_VAMPIRES_MBI",
        "20240729_VAMPIRES_MBI",
    ]
    ## Plot and save

    names = ["F610", "F670", "F720", "F760"]
    psfs = [fits.getdata(paths.data / f"VAMPIRES_{filt}_synthpsf.fits") for filt in names]
    vamp_psf = sum(psfs) # add together PSFs
    vamp_psf /= sum(vamp_psf) # normalize PSF kernel

    for i, folder in enumerate(tqdm.tqdm(folders)):
        # load data
        with fits.open(
            paths.data
            / folder
            / "diskmap"
            / f"{folder}_HD169142_diskmap_deprojected.fits"
        ) as hdul:
            deproj_cube = hdul[0].data

        radii = frame_radii(deproj_cube)

        # warp to polar coordinates
        # if "VAMPIRES" in folder:
        #     frame = convolve(deproj_cube, vamp_psf)
        # else:
        frame = convolve(deproj_cube, kernels.Gaussian2DKernel(1 / (2 * np.sqrt(2 * np.log(2)))))

        max_rad = deproj_cube.shape[-2] // 2

        polar_frame, polar_settings = pt.convertToPolarImage(
            np.rot90(np.nan_to_num(frame * radii**2)),
            angleSize=360//_DEG_PER_PIXEL,  # 5 degree per bin
            initialRadius=0,
            finalRadius=max_rad,
            radiusSize=max_rad,
            order=3,
        )
        polar_frame = np.transpose(polar_frame)

        outname = paths.data / folder / f"{folder}_HD169142_Qphi_polar.fits"
        fits.writeto(outname, polar_frame, overwrite=True)
