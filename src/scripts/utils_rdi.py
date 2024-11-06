from winnie_pcrdi import crop_data, dist_to_pt, rdi_residuals, sigma_clipped_axis_nanmean_general
import numpy as np

def rdi(sci_cube, ref_cube, pas, rmin, rmax, half_width):
    # minimum sep in pixels for optimization zone
    # typically IWA + a few pixels
    rmin = 69 // 5.9 + 4
    # maximum sep in pixels for optimization zone
    rmax = 100
    # image half-width for cropping
    half_width = 200

    # crop science and reference cube
    new_shape = (half_width*2 + 1, half_width*2 + 1)
    hcube, cent = crop_data(sci_cube, new_shape)
    hcube_ref = crop_data(ref_cube, new_shape)[0]

    # Build opt/sub zones for PSF subtraction
    nT,nL,ny,nx = hcube.shape
    rmap = dist_to_pt(cent, nx=nx, ny=ny, dtype=hcube.dtype)
    fovmask = np.all(np.nan_to_num(hcube) != 0., axis=(0,1)) & np.all(np.nan_to_num(hcube_ref) != 0., axis=(0,1))

    # Optimization and subtraction zones. Using just one zone here.
    optzones = np.array([fovmask & (rmap > rmin) & (rmap <= rmax)])
    subzones = np.array([fovmask])

    hcube_res_derot = rdi_residuals(hcube, hcube_ref, optzones, subzones, hcube_css_est=None, show_progress=True, parangs=-pas, cent=cent)
    imcube0 = sigma_clipped_axis_nanmean_general(hcube_res_derot, n=3, axis=0, fast=True)
    return imcube0

