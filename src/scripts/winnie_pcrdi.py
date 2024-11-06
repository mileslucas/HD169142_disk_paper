from IPython.display import display, HTML
import scipy.linalg as linalg
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects
from matplotlib import animation
from joblib import Parallel, delayed
from copy import copy, deepcopy
from astropy import units as u
from astropy.io import fits
from scipy import ndimage, signal
import glob
from tqdm.auto import tqdm
from astropy.convolution import Tophat2DKernel, Gaussian2DKernel, convolve
import math
import lmfit


target_defaults = {}


def rdi_residuals(hcube, hcube_ref, optzones, subzones, hcube_css_est=None, ref_mask=None, show_progress=False, parangs=None, cent=None, 
                  objective=False, zero_nans=False, use_gpu=False, ncores=-2, return_coeffs=False, coeffs_in=None, return_psf_model=False,
                  pad_before_derot=False, opt_smoothing=False, opt_smoothing_fn=None, opt_smoothing_kwargs=None):
    """
    Performs RDI PSF subtraction using LOCI (Lafreniere et al. 2007), optionally allowing a 'constraint' on the circumstellar 
    signal contained in the data to mitigate oversubtraction (Lawson et al. 2022).
    ___________
    Parameters:
    
        hcube: ndarray
            4D image array to be PSF-subtracted; shape of (nT, nL, ny, nx) where nT is the number of exposures/integrations, 
            nL is the number of wavelengths, and ny & nx are the number of pixels in each spatial dimension.
            
        hcube_ref: ndarray
            4D or 5D image array; for 4D, shape should be (nT_ref, nL, ny, nx) where nT_ref is the number of reference
            exposures/integrations, and the remaining axes match those of hcube. For 5D, the shape should be 
            (nT, nT_ref, nL, ny, nx), in which a distinct set of references is provided for each entry of hcube. 
            In this case, the PSF model for exposure Ti of hcube would be constructed from hcube_ref[Ti]. 
            
        optzones: ndarray
            3D boolean array; for each slice, the target and reference images will be compared over any pixels with
            a value of True. The resulting coefficients will be used to perform PSF-subtraction over the region
            indicated by the corresponding entry in subzones.
            
        subzones: ndarray
            3D boolean array; for each slice, PSF subtraction will be performed only on pixels with
            a value of True. 
            
    _________
    Optional:
    
        hcube_css_est: ndarray
            4D array; same shape as hcube. hcube_css_est should provide an estimate of the circumstellar signal 
            in hcube, rotated to the appropriate parangs and convolved with the appropriate PSF.
            
        ref_mask: ndarray
            2D boolean array of shape (len(optzones), nT_ref) that indicates which reference images should be considered
            for which optimization regions. E.g., if ref_mask[i,j] is False, then for the ith optimization zone (optzones[i]),
            the jth reference image (hcube_ref[j]) will NOT be used for construction of the PSF model. This can be useful if
            some reference exposures have anomalous features that make them problematic for some regions while still being
            suitable for others; e.g., an image with a bright background source near the edge of the FOV may still be useful
            for nulling the PSF near the inner working angle.
            
        show_progress: bool
            If True, a status bar will be displayed for the major components of PSF subtraction.
            
        parangs: ndarray
            1D array giving the parallactic angles (in degrees) of each exposure in hcube. If provided, the output 
            residual hypercube will be derotated accordingly.
            
        cent: ndarray
            The cartesian pixel coordinate (x,y) corresponding to the central star's position in hcube for the purpose 
            of derotation.
            
        objective: bool
            If True, the output array will be (hcube - hcube_css_est) - hcube_psfmodel, where hcube_psfmodel is 
            constructed by comparing hcube_ref to (hcube - hcube_css_est). If False, the output array will simply
            be hcube - hcube_psfmodel. If hcube_css_est is None, then setting this will have no effect. 
            
        zero_nans: bool
            If True, any nans in hcube or hcube_css_est will be replaced with zeros for the procedure.
            
        use_gpu: bool
            If True, use faster GPU-based CuPy routines throughout.
            
        ncores: int
            The number of processor cores to use. Default value of -2 uses all but one available core.
            
        return_psf_model: bool
            If True, the PSF-model hcube matching hcube in shape is returned instead of the residuals hcube. Will not be
            derotated (even if parangs is specified).

        pad_before_derot: bool
            If True, prior to derotation, the residuals are padded to sufficient size to avoid loss of pixels. Note: 
            output dimensions will not match that of the input when this option is used. 

        return_coeffs: bool
            If True, returns only the array of PSF model coefficients.
        
        return_psf_model: bool
            If True, returns only the PSF model hypercube with the same shape as the data (hcube).

        coeffs_in: ndarray
            If provided, these coefficients will be used to construct the PSF model instead of computing coefficients.

        opt_smoothing: bool
            If True, the reference coefficient calculation uses smoothed versions of the target and reference sequences.
            May be helpful when working with data that may have uncorrected cosmic rays, etc.
            
        opt_smoothing_fn: callable
            If opt_smoothing==True, this argument indicates the function with which to smooth the sequences. This should
            be a function that takes a hypercube along with some keyword arguments and returns a smoothed hypercube, 
            i.e.: hcube_filt = opt_smoothing_fn(hcube, **opt_smoothing_kwargs).   
            Defaults to median_filter_sequence when opt_smoothing is set.
        
        opt_smoothing_kwargs: dict
            If opt_smoothing==True, keyword arguments to pass to opt_smoothing_fn when it is called.
    ________
    Returns:
        Default:
            hcube_res: ndarray
                4D array of PSF-subtracted residuals (derotated if 'parangs' was specified).

        if pad_before_derot:
            (hcube_res, cent_pad): tuple
                4D array of PSF-subtracted residuals and the center location with consideration for the added padding.
    """
    nT = hcube.shape[0]
    
    if isNone(hcube_css_est):
        hcube_sub = hcube
    else:
        hcube_sub = hcube - hcube_css_est

    if zero_nans: 
        hcube_sub = np.nan_to_num(hcube_sub)
        
    if opt_smoothing:
        if isNone(opt_smoothing_kwargs):
            footprint = np.zeros((3,3), dtype='bool')
            footprint[1,:] = True
            footprint[:,1] = True
            opt_smoothing_kwargs = dict(footprint=footprint)
        if isNone(opt_smoothing_fn):
            opt_smoothing_fn = median_filter_sequence
        hcube_opt, hcube_ref_opt = opt_smoothing_fn(hcube_sub, **opt_smoothing_kwargs), opt_smoothing_fn(hcube_ref, **opt_smoothing_kwargs)
        
    else:
        hcube_opt, hcube_ref_opt = hcube_sub, hcube_ref
    
    if hcube.ndim == hcube_ref.ndim:
        if isNone(coeffs_in):
            coeffs = compute_rdi_coefficients(hcube_opt, hcube_ref_opt, optzones, show_progress=show_progress, ref_mask=ref_mask)
        else:
            coeffs = coeffs_in
        if return_coeffs:
            return coeffs
        psf_model = reconstruct_psf_model(hcube_ref, coeffs, subzones, show_progress=show_progress, use_gpu=use_gpu, ncores=ncores)

    else:
        coeffs = []
        for Ti in range(nT):
            coeffs.append(compute_rdi_coefficients(hcube_opt[[Ti]], hcube_ref_opt[Ti], optzones, show_progress=show_progress, ref_mask=ref_mask))
        if return_coeffs:
            return coeffs
        psf_model = np.array([reconstruct_psf_model(hcube_ref[Ti], coeffs[Ti], subzones, show_progress=show_progress, use_gpu=use_gpu, ncores=ncores)[0]
                              for Ti in range(nT)])
        
    if return_psf_model:
        return psf_model
    
    if objective:
        hcube_res = hcube_sub - psf_model
    else:
        hcube_res = hcube - psf_model
        
    if not isNone(parangs):
        if pad_before_derot:
            hcube_res = pad_and_rotate_hypercube(hcube_res, -parangs, cent=cent, ncores=ncores, use_gpu=use_gpu, cval0=np.nan)
        else:
            hcube_res = rotate_hypercube(hcube_res, -parangs, cent=cent, ncores=ncores, use_gpu=use_gpu, cval0=np.nan)
    if use_gpu:
        free_gpu()
    return hcube_res


def adi_residuals(hcube, parangs, optzones, subzones, fwhm_arr, hcube_css_est=None, nfwhm=1.0, r_opt=None,
                  cent=None, ncores=-2, use_gpu=False, no_derot=False, show_progress=False, pad_before_derot=False):
    """
    Most of these arguments are explained in the documentation for rdi_residuals. Exceptions:

    """
    
    if isNone(hcube_css_est):
        hcube_sub = hcube
    else:
        hcube_sub = hcube - hcube_css_est
        
    if isNone(r_opt):
        if isNone(cent):
            cent = (np.array(hcube.shape[-2:][::-1])-1)/2.
        rmap = dist_to_pt(cent, nx=hcube.shape[-1], ny=hcube.shape[-2], dtype=hcube.dtype)
        r_opt = np.array([np.min(rmap[optzone]) for optzone in optzones])
        
    coeffs = compute_adi_coefficients(hcube_sub, parangs, optzones, r_opt,
                                      nfwhm, fwhm_arr, show_progress=show_progress)

    psf_model = reconstruct_psf_model(hcube_sub, coeffs, subzones, show_progress=show_progress, use_gpu=use_gpu, ncores=ncores)
    hcube_res = hcube - psf_model
    if not no_derot:
        if pad_before_derot:
            hcube_res = pad_and_rotate_hypercube(hcube_res, -parangs, cent=cent, ncores=ncores, use_gpu=use_gpu, cval0=np.nan)
        else:
            hcube_res = rotate_hypercube(hcube_res, -parangs, cent=cent, ncores=ncores, use_gpu=use_gpu, cval0=np.nan)
    return hcube_res


def compute_rdi_coefficients(hcube, hcube_ref, optzones, show_progress=False, ref_mask=None):
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ModuleNotFoundError:
            print('tqdm module not found!\n'
                  'To show progress bar ("show_progress = True")\n'
                  'install tqdm (e.g. "pip install tqdm").\n'
                  'Proceeding without progress bar . . .')
            show_progress = False
            
    nT, nL, ny, nx = hcube.shape  # N_theta by N_lambda by N_y by N_x
    nT_ref = hcube_ref.shape[0]
    nR = len(optzones) # N_regions
    coeff_hcube = np.zeros((nL, nR, nT, nT_ref))  # Array for storing coefficients
    R_iterator = tqdm(range(nR), desc='Coefficient calculation (regions)', leave=False) if show_progress else range(nR)
    L_iterator = tqdm(range(nL), desc='Coefficient calculation (wavelengths)', leave=False) if (show_progress and nL>1) else range(nL)
    T_iterator = tqdm(range(nT), desc='Coefficient calculation (exposures)', leave=False) if show_progress else range(nT)
    for Ri in R_iterator:  # Outermost loop over subsections; this iteration order ends up being more time efficient here:
        opt_i = optzones[Ri]  # opt_i is the ny by nx boolean array indicating which pixels are in the region.
        tararrs = hcube[:, :, opt_i].copy() # This turns our masked nT*nL*ny*nx array into an nT*nL*nP array, where nP is the number of pixels in the optimization region.
        refarrs = hcube_ref[:, :, opt_i].copy()  # This turns our masked nT_ref*nL*ny*nx array into an nT_ref*nL*nP array, where nP is the number of pixels in the optimization region.
        if not isNone(ref_mask):
            refarrs = refarrs[ref_mask[Ri]]
        optmats = refarrs.transpose((1, 0, 2)) @ tararrs.transpose((1, 2, 0))# Using python 3.6+ magic, carries out matrix inversion of all wavelength channels at once, giving a matrix of shape (nL, nT_ref, nT)
        refmats = refarrs.transpose((1, 0, 2)) @ refarrs.transpose((1, 2, 0))# Using python 3.6+ magic, carries out matrix inversion of all wavelength channels at once, giving a matrix of shape (nL, nT_ref, nT_ref)
        for Li in L_iterator:  # Second loop over wavelengths:
            optmat = optmats[Li]  # The (nT_ref, nT) matrix for this wavelength
            refmat = refmats[Li]  # The (nT_ref, nT_ref) matrix for this wavelength
            lu, piv = linalg.lu_factor(refmat)  # Since we aren't excluding frames as in ADI/SDI, we just need to run this once per wavelength.
            for Ti in T_iterator:  # Final loop over par ang / time axis (individual science exposures):
                tararr = optmat[:,Ti]  # 1d vector of length equal to nT_ref
                if not isNone(ref_mask):
                    coeff_hcube[Li, Ri, Ti, ref_mask[Ri]] = linalg.lu_solve((lu, piv), tararr) # Gets coefficients and places them into the appropriate positions in the coefficient array
                else:
                    coeff_hcube[Li, Ri, Ti] = linalg.lu_solve((lu, piv), tararr) # Gets coefficients and places them into the appropriate positions in the coefficient array
    return coeff_hcube


def compute_adi_coefficients(hcube, parangs, optzones, r1, nfwhm, fwhm_array, show_progress=False):
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ModuleNotFoundError:
            print('tqdm module not found!\n'
                  'To show progress bar ("show_progress = True")\n'
                  'install tqdm (e.g. "pip install tqdm").\n'
                  'Proceeding without progress bar . . .')
            show_progress = False
            
    nT, nL, ny, nx = hcube.shape  # N_theta by N_lambda by N_y by N_x
    nR = len(optzones) # N_regions
    coeff_hcube = np.zeros((nL, nR, nT, nT))  # Array for storing coefficients
    R_iterator = tqdm(range(nR), desc='Coefficient calculation (regions)', leave=False) if show_progress else range(nR)
    L_iterator = tqdm(range(nL), desc='Coefficient calculation (wavelengths)', leave=False) if (show_progress and nL>1) else range(nL)
    T_iterator = tqdm(range(nT), desc='Coefficient calculation (exposures)', leave=False) if show_progress else range(nT)
    min_disps = nfwhm * fwhm_array
    for Ri in R_iterator:  # Outermost loop over subsections; this iteration order ends up being more time efficient here:
        opt_i = optzones[Ri]  # opt_i is the ny by nx boolean array indicating which pixels are in the region.
        optarrs = hcube[:, :, opt_i].copy() # This turns our masked nT*nL*ny*nx array into an nT*nL*nP array, where nP is the number of pixels in the optimization region.
        optmats = optarrs.transpose((1, 0, 2)) @ optarrs.transpose((1, 2, 0))# Using python 3.6+ magic, carries out matrix inversion of all 22 wavelength channels at once, giving a matrix of shape (nL, nT_ref, nT)
        for Li in L_iterator:  # Second loop over wavelengths:
            min_disp = min_disps[Li]  # The minimum displacement in pixels for this wavelength
            optmat = optmats[Li]  # The (nT,nT) matrix for this wavelength
            for Ti in T_iterator:  # Final loop over par ang / time axis (individual science exposure cubes):
                dtheta = parangs - parangs[Ti]  # Difference between parang of each frame and the target frame
                disp = ang_displacement(r1[Ri], dtheta)  # Corresponding pixel displacement at the inner edge of the optimization region.
                valid_dtheta = disp > min_disp  # Mask of shape (nT,) indicating which frames are valid references. Choosing > rather than >= just ensures a non-trivial result in the case for min_disp = 0 (where >= would include the tar image)
                if np.sum(valid_dtheta) == 0.: continue # Move on if no valid frames are available for this subsection.
                optmat_v1 = optmat[valid_dtheta]
                tararr = optmat_v1[:, Ti] # 1d vector of length equal to the number of "valid" images in sequence
                refmat = optmat_v1[:, valid_dtheta]
                lu, piv = linalg.lu_factor(refmat)
                coeff_hcube[Li, Ri, Ti, valid_dtheta] = linalg.lu_solve((lu, piv), tararr)
    return coeff_hcube


def reconstruct_psf_model(hcube_ref, coeffs, subzones, show_progress=False, use_gpu=False, ncores=-2):
    if use_gpu:
        hcube_psfmodel = reconstruct_psf_model_gpu(hcube_ref, coeffs, subzones, show_progress=show_progress)
    else:
        hcube_psfmodel = reconstruct_psf_model_cpu(hcube_ref, coeffs, subzones, show_progress=show_progress, ncores=ncores)
    return hcube_psfmodel


def reconstruct_psf_model_cpu(hcube_ref, coeffs, subzones, show_progress=False, ncores=-2):
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ModuleNotFoundError:
            print('tqdm module not found!\n'
                  'To show progress bar ("show_progress = True")\n'
                  'install tqdm (e.g. "pip install tqdm").\n'
                  'Proceeding without progress bar . . .')
            show_progress = False
    _, _, ny, nx = hcube_ref.shape # Number of: images, wavelengths, y-pixels, and x-pixels
    nL, nR, nT, _ = coeffs.shape
    hcube_psfmodel = np.zeros((nT, nL, ny, nx)) + np.nan # Array in which to place reconstructed model
    R_iterator = tqdm(range(nR), desc='PSF model reconstruction (regions)', leave=False) if show_progress else range(nR) # Iterator for regions
    T_iterator = tqdm(range(nT), desc='PSF model reconstruction (exposures)', leave=False) if show_progress else range(nT) #  Iterator for images
    for Ri in R_iterator:
        sub_i = subzones[Ri] # An ny by nx boolean mask indicating which pixels are being considered
        imvals = hcube_ref[:, :, sub_i].T.copy() # Fetching the pixels in the subzone, dimensions of nT_ref x nL x npx
        nT_results = Parallel(n_jobs=ncores, prefer='threads')(delayed(np.sum)(coeffs[np.newaxis, :, Ri, Ti].copy()*imvals, -1) for Ti in T_iterator)
        for Ti in range(nT): # Iterate over target images, building the PSF model (in the subzone) for each.
            hcube_psfmodel[Ti, :, sub_i] = nT_results[Ti] # Multiply images by the coefficients, then sum along the nT axis to get the model values.
    return hcube_psfmodel


def reconstruct_psf_model_gpu(hcube_ref, coeffs, subzones, show_progress=False):
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ModuleNotFoundError:
            print('tqdm module not found!\n'
                  'To show progress bar ("show_progress = True")\n'
                  'install tqdm (e.g. "pip install tqdm").\n'
                  'Proceeding without progress bar . . .')
            show_progress = False
            
    cp_hcube_ref = cp.array(hcube_ref)
    cp_coeffs = cp.array(coeffs)
    cp_subzones = cp.array(subzones)
    
    _, _, ny, nx = cp_hcube_ref.shape # Number of: images, wavelengths, y-pixels, and x-pixels
    nL, nR, nT, _ = cp_coeffs.shape
    
    hcube_psfmodel = cp.zeros((nT, nL, ny, nx))+np.nan # Array in which to place reconstructed model
    R_iterator = tqdm(range(nR), desc='PSF model reconstruction (regions)', leave=False) if show_progress else range(nR) # Iterator for regions
    T_iterator = tqdm(range(nT), desc='PSF model reconstruction (exposures)', leave=False) if show_progress else range(nT) #  Iterator for images
    for Ri in R_iterator:
        sub_i = cp_subzones[Ri] # An ny by nx boolean mask indicating which pixels are being considered
        imvals = cp_hcube_ref[:, :, sub_i].transpose((1,0,2)) # Fetching the pixels in the subzone, dimensions of nT_ref x nL x npx
        for Ti in tqdm(T_iterator, desc='Target image', leave=False) if show_progress else T_iterator: # Iterate over target images, building the PSF model (in the subzone) for each.
            cvals = cp_coeffs[:, Ri, Ti, :, cp.newaxis] # Fetch the appropriate coefficients
            hcube_psfmodel[Ti, :, sub_i] = cp.sum(cvals*imvals, axis=1) # Multiply images by the coefficients, then sum along the nT axis to get the model values.
    hcube_psfmodel_np = cp.asnumpy(hcube_psfmodel) # Convert output back to numpy
    hcube_psfmodel, cp_hcube_ref, cp_coeffs, cp_subzones, cvals = free_gpu(hcube_psfmodel, cp_hcube_ref, cp_coeffs, cp_subzones, cvals) # Explicitly clear VRAM for cupy arrays
    return hcube_psfmodel_np



def rdi_greeds(hcube, hcube_ref, parangs, optzones, subzones, cent=None, ncores=-2, use_gpu=False, niter=10, show_progress=True):
    """
    Arguments as for rdi_residuals, except:

    niter: int
        The number of GReeDS iterations to carry out.

    ________
    Returns:
        greeds_res: ndarray
            Array of shape (niter, nL, ny, nx) — the GreeDS residual cube resulting from each iteration. 
            Average along axis 1 to get wavelength-collapsed images.
    """
    nT,nL,ny,nx = hcube.shape
    if isNone(cent):
        cent = (np.array([nx, ny])-1.)/2.
        
    # Determine a reasonable cropped size and then crop the data for iteration:
    rmap = dist_to_pt(cent, nx=nx, ny=ny, dtype=hcube.dtype)
    crop_shape = np.repeat(int(np.ceil(2*(np.sqrt(2)*np.max(rmap[np.any(optzones, axis=0)]))))+5, 2)
    cr_hcube, cr_cent = crop_data(hcube, crop_shape, cent)
    cr_hcube_ref, cr_optzones, cr_subzones = [crop_data(i, crop_shape, cent)[0] for i in [hcube_ref, optzones, subzones]]
    cr_ny, cr_nx = cr_hcube.shape[-2:]
    
    greeds_res = np.zeros((niter, nL, cr_ny, cr_nx), dtype=hcube.dtype)
    n_iterator = tqdm(range(niter)) if show_progress else range(niter)
    for i in n_iterator:
        if i == 0:
            hcube_css = np.zeros_like(cr_hcube)
        else:
            hcube_css = imcube_to_adi_hcube(css*(css>0), parangs, cent=cr_cent, use_gpu=use_gpu, ncores=ncores)
        res_hcube = rdi_residuals(cr_hcube, cr_hcube_ref, cr_optzones, cr_subzones, parangs=parangs, cent=cr_cent,
                                  use_gpu=use_gpu, ncores=ncores, hcube_css_est=hcube_css, show_progress=show_progress)
        css = sigma_clipped_axis_nanmean_general(res_hcube, axis=0, fast=True)
        greeds_res[i] = css
        
    # Now: get PSF model coeffs for the final iterate
    hcube_css = imcube_to_adi_hcube(css*(css>0), parangs, cent=cr_cent, use_gpu=use_gpu, ncores=ncores)
    coeffs = rdi_residuals(cr_hcube, cr_hcube_ref, cr_optzones, cr_subzones, parangs=parangs, cent=cr_cent,
                           use_gpu=use_gpu, ncores=ncores, hcube_css_est=hcube_css, show_progress=show_progress,
                           return_coeffs=True)
    
    # Run RDI on fullframe data using these coefficients
    hcube_residuals = rdi_residuals(hcube, hcube_ref, optzones, subzones, coeffs_in=coeffs, 
                              parangs=parangs, cent=cent, use_gpu=use_gpu, ncores=ncores,
                              show_progress=show_progress)
    
    imcube_greeds = np.nanmedian(hcube_residuals, axis=0)
    return imcube_greeds, greeds_res


def adi_greeds(hcube, parangs, optzones, subzones, nfwhm, fwhm_arr, cent=None, r_opt=None, ncores=-2, use_gpu=False, niter=10, show_progress=True):
    nT,nL,ny,nx = hcube.shape
    if isNone(cent):
        cent = (np.array([nx, ny])-1.)/2.
    if isNone(r_opt):
        rmap = dist_to_pt(cent, nx=nx, ny=ny, dtype=hcube.dtype)
        r_opt = np.array([np.min(rmap[optzone]) for optzone in optzones])
    greeds_res = np.zeros((niter, nL, ny, nx), dtype=hcube.dtype)
    n_iterator = tqdm(range(niter)) if show_progress else range(niter)
    for i in n_iterator:
        if i == 0:
            hcube_css = np.zeros_like(hcube)
        else:
            hcube_css = imcube_to_adi_hcube(np.nan_to_num(css*(css>0)), parangs, cent=cent, use_gpu=use_gpu, ncores=ncores)
        res_hcube = adi_residuals(hcube, parangs, optzones, subzones, fwhm_arr, hcube_css_est=hcube_css, nfwhm=nfwhm, r_opt=r_opt,
                                  cent=cent, ncores=ncores, use_gpu=use_gpu, show_progress=show_progress) # Residuals
        css = np.nanmean(res_hcube, axis=0)
        greeds_res[i] = css
    return greeds_res


def setup_display(width=95, fontsize=18):
    """
    Sets window width and markdown fontsize for Jupyter notebook. Width is % of window.
    """
    display(HTML("<style>.container { width:"+str(width)+"% !important; }</style>"))
    display(HTML("<style>.rendered_html { font-size: "+str(fontsize)+"px; }</style>"))
    return None


def source(fn):
    import inspect
    print(inspect.getsource(fn))
    return None


def dist_to_pt(pt, nx=201, ny=201, dtype=float):
    """
    Returns a square distance array of size (ny,nx), 
    where each pixel corresponds to the euclidean distance
    of that pixel from "pt".
    """
    xaxis = np.arange(0, nx, dtype=dtype)-pt[0]
    yaxis = np.arange(0, ny, dtype=dtype)-pt[1]
    return np.sqrt(xaxis**2 + yaxis[:, np.newaxis]**2)


def rotate_image_cpu(im, angle, cent=None, new_cent=None, cval0=np.nan, prop_threshold=1e-6):
    """
    Rotates im by angle "angle" in degrees using CPU operations. Avoids "mixing" exact zero values,
    which should functionally be treated as nans. If cent is provided, rotates about cent. 
    Otherwise, uses ndimage's rotate (which is a bit faster) to rotate about the geometric center.
    """
    from scipy import ndimage
    if angle == 0.:
        return im.copy()
    geom_cent = (np.array(im.shape[-2:][::-1])-1.)/2.
    if isNone(cent) or np.all(cent == geom_cent):
        im_out = propagate_nans_in_spatial_operation(im, ndimage.rotate, fn_args=[angle],
                                                     fn_kwargs=dict(axes=(-2, -1), reshape=False, cval=cval0),
                                                     fn_nan_kwargs=dict(axes=(-2, -1), reshape=False, prefilter=False),
                                                     prop_threshold=prop_threshold, prop_zeros=True)
    else:
        im_out = propagate_nans_in_spatial_operation(im, rotate_about_pos, fn_args=[cent, angle],
                                                     fn_kwargs=dict(cval=cval0),
                                                     fn_nan_kwargs=dict(cval=0, prefilter=False),
                                                     prop_threshold=prop_threshold, prop_zeros=True)
    return im_out


def rotate_image_gpu(im0, angle, cent=None, cval0=np.nan):
    """
    Rotates im by angle "angle" in degrees using GPU operations. Avoids "mixing" exact zero values, which should functionally be treated as nans.
    If cent is provided, rotates about cent. Otherwise, uses CuPy's version of scipy.ndimage's rotate (which is a bit faster) to rotate about the
    geometric center.
    """
    if angle == 0.:
        return im0.copy()
    im = cp.asarray(im0)
    nans = cp.isnan(im)
    zeros = im == 0.
    any_zeros = cp.any(zeros)
    any_nans = cp.any(nans)
    geom_cent = (np.array(im.shape[-2:][::-1])-1.)/2.
    if isNone(cent) or np.all(cent == geom_cent):
        if any_nans:
            rot_im = cp_ndimage.rotate(cp.where(nans, 0., im), angle, axes=(-2, -1), reshape=False, cval=cval0)
        else:
            rot_im = cp_ndimage.rotate(im, angle, axes=(-2, -1), reshape=False, cval=cval0)
        if any_zeros:
            rot_zeros = cp_ndimage.rotate(zeros.astype(float), angle, axes=(-2, -1),  prefilter=False, reshape=False)
            rot_im = cp.where(rot_zeros>0., 0., rot_im)
        if any_nans:
            rot_nans = cp_ndimage.rotate(nans.astype(float), angle, axes=(-2, -1),  prefilter=False, reshape=False)
            rot_im = cp.where(rot_nans>0., cp.nan, rot_im)
    else:
        if any_nans:
            rot_im = rotate_about_pos_gpu(cp.where(nans, 0., im), cent, angle, cval=cval0)
        else:
            rot_im = rotate_about_pos_gpu(im, cent, angle, cval=cval0)
        if any_zeros:
            rot_zeros = rotate_about_pos_gpu(zeros.astype(float), cent, angle,  prefilter=False)
            rot_im = cp.where(rot_zeros>0., 0., rot_im)
        if any_nans:
            rot_nans = rotate_about_pos_gpu(nans.astype(float), cent, angle,  prefilter=False)
            rot_im = cp.where(rot_nans>0., cp.nan, rot_im)
    return cp.asnumpy(rot_im)


def rotate_about_pos_gpu(im, pos, angle, cval=np.nan, order=3, mode='constant', prefilter=True):
    ny, nx = im.shape[-2:]
    nd = cp.ndim(im)
    xg0, yg0 = cp.meshgrid(cp.arange(nx, dtype=cp.float64), cp.arange(ny, dtype=cp.float64))
    
    xg,yg = xy_polar_ang_displacement_gpu(xg0-pos[0], yg0-pos[1], angle)
    xg += pos[0]
    yg += pos[1]
    
    if nd == 2:
        im_rot = cp_ndimage.map_coordinates(im, cp.array([yg,xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
    else:
        nI = int(cp.prod(cp.array(im.shape[:-2])))
        im_reshaped = im.reshape((nI, ny, nx))
        im_rot = cp.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_rot[i] = cp_ndimage.map_coordinates(im_reshaped[i], cp.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
        im_rot = im_rot.reshape((*im.shape[:-2], ny, nx))
    xg, yg, xg0, yg0 = free_gpu(xg, yg, xg0, yg0)
    return im_rot


def rotate_hypercube(hcube, angles, cent=None, new_cent=None, ncores=-2, use_gpu=False, cval0=0.):
    """
    Rotates an N-dimensional array, 'hcube', where the final two axes are assumed to be cartesian y and x 
    and where 'angles' is an array of angles (in degrees) matching the length of the first dimension.
    
    E.g., for a sequence of nT images having shape (ny,nx), hcube should have shape (nT,ny,nx) and angles should have shape (nT,)
    
    For a sequence of nT IFS image cubes each having nL wavelength images of shape (ny,nx), hcube should have shape (nT, nL, ny, nx)
    """
    if use_gpu:
        rot_hcube = np.stack([rotate_image_gpu(imcube, angle, cval0=cval0, cent=cent, new_cent=new_cent) for imcube, angle in zip(hcube, angles)])
    else:
        rot_hcube = np.stack(Parallel(n_jobs=ncores, prefer='threads')(delayed(rotate_image_cpu)(imcube, angle, cval0=cval0, cent=cent, new_cent=new_cent) for imcube, angle in zip(hcube, angles)))
    return rot_hcube


def pad_and_rotate_hypercube(hcube, angles, cent=None, ncores=-2, use_gpu=False, cval0=np.nan):
    """
    Like rotate_hypercube, but pads the images first to avoid loss of pixels. Returns the rotated 
    hypercube and the new center of the padded hypercube.
    """
    ny, nx = hcube.shape[-2:]
    if isNone(cent):
        cent = (np.array([nx, ny])-1.)/2.
    dxmin, dxmax = np.array([0, nx]) - cent[0]
    dymin, dymax = np.array([0, ny]) - cent[1]
    corner_coords = np.array([[dxmax, dymax],
                              [dxmax, dymin],
                              [dxmin, dymin],
                              [dxmin, dymax]])
    uni_angs = np.unique(angles)
    derot_corner_coords = np.vstack([np.array(xy_polar_ang_displacement(*corner_coords.T, -ang)).T for ang in uni_angs])
    dxmin_pad, dymin_pad = (np.ceil(np.abs(np.min(derot_corner_coords, axis=0) - np.array([dxmin, dymin])))).astype(int)
    dxmax_pad, dymax_pad = (np.ceil(np.abs(np.max(derot_corner_coords, axis=0) - np.array([dxmax, dymax])))).astype(int)
    hcube_pad = np.pad(hcube.copy(), [*[[0,0] for i in range(hcube.ndim-2)], [dymin_pad, dymax_pad], [dxmin_pad, dxmax_pad]], constant_values=np.nan)
    cent_pad = cent + np.array([dxmin_pad, dymin_pad])
    hcube_pad_rot = rotate_hypercube(hcube_pad, angles, cent=cent_pad, ncores=ncores, use_gpu=use_gpu, cval0=cval0)
    return hcube_pad_rot, cent_pad


def rotate_about_pos(im, pos, angle, new_cent=None, cval=np.nan, order=3, mode='constant', prefilter=True):
    ny, nx = im.shape[-2:]
    nd = np.ndim(im)
    xg0, yg0 = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64))
    
    if not isNone(new_cent):
        xg0 -= (new_cent[0]-pos[0])
        yg0 -= (new_cent[1]-pos[1])
    
    xg,yg = xy_polar_ang_displacement(xg0-pos[0], yg0-pos[1], angle)
    xg += pos[0]
    yg += pos[1]

    if nd == 2:
        im_rot = ndimage.map_coordinates(im, np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
    else:
        nI = np.product(im.shape[:-2])
        im_reshaped = im.reshape((nI, ny, nx))
        im_rot = np.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_rot[i] = ndimage.map_coordinates(im_reshaped[i], np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
        im_rot = im_rot.reshape((*im.shape[:-2], ny, nx))
    return im_rot


def xy_polar_ang_displacement(x, y, dtheta):
    """
    Rotates cartesian coordinates x and y by angle dtheta (deg) about (0,0).
    """
    r = np.sqrt(x**2+y**2)
    theta = np.rad2deg(np.arctan2(y,x))
    new_theta = np.deg2rad(theta+dtheta)
    newx,newy = r*np.cos(new_theta),r*np.sin(new_theta)
    return newx,newy


def xy_polar_ang_displacement_gpu(x, y, dtheta):
    r = cp.sqrt(x**2+y**2)
    theta = cp.rad2deg(cp.arctan2(y,x))
    new_theta = cp.deg2rad(theta+dtheta)
    newx,newy = r*cp.cos(new_theta),r*cp.sin(new_theta)
    return newx,newy


def free_gpu(*args):
    N = len(args)
    args = list(args)
    for i in range(N):
        args[i] = None
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    if N <= 1:
        return None
    return args


def isNone(arg):
    """
    Just a quick convenience/shorthand function.
    "if isNone(x)" works for any x, whereas "if x == None"
    will sometimes cause a crash (e.g., if x is a numpy array).
    """
    return isinstance(arg, type(None))


def numpy_to_gpu_nanmedian(x, axis=None):
    x_cp = cp.array(x)
    med_cp = gpu_nanmedian(x_cp, axis=axis)
    
    med_numpy = cp.asnumpy(med_cp)
    x_cp, med_cp = free_gpu(x_cp, med_cp)
    return med_numpy


def gpu_nanmedian(x, axis=None):
    axes = np.arange(x.ndim)
    preserved_axes = axes[~np.isin(axes, np.asarray(axis))]
    
    in_shape = np.asarray(x.shape)
    out_shape = in_shape[preserved_axes]
    out = cp.zeros(tuple(out_shape), dtype=x.dtype)
    avail_mem = gpu.mem_info[0]+(cp.get_default_memory_pool().total_bytes() - cp.get_default_memory_pool().used_bytes())
    if avail_mem < (x.nbytes*6):
        nparts = int(np.ceil(x.nbytes*6 / avail_mem))
        sorted_axes = np.argsort(in_shape)
        split_axis = sorted_axes[np.isin(sorted_axes, preserved_axes)][-1]
        nsplit = in_shape[split_axis]
        split_inds = cp.array_split(cp.arange(nsplit), nparts)
        for inds in split_inds:
            out[index_axis(inds, np.where(preserved_axes == split_axis)[0][0])] = cp.nanmedian(x[index_axis(inds, split_axis)], axis=axis)
            free_gpu()
    else:
        cp.nanmedian(x, axis=axis, out=out)
        free_gpu()
    return out


def index_axis(index, axis):
    return (slice(None), )*axis + (index, )


def median_combine_sequence(hcube, use_gpu=False, axis=0):
    if use_gpu:
        out = numpy_to_gpu_nanmedian(hcube, axis=axis)
    else:
        out = np.nanmedian(hcube, axis=axis)
    return out


def quick_implot(im, clim=None, clim_perc=[1.0, 99.0], cmap=None,
                 show_ticks=False, lims=None, ylims=None,
                 norm=mpl.colors.Normalize, norm_kwargs={},
                 figsize=None, panelsize=[5,5], fig_and_ax=None, extent=None,
                 show=True, tight_layout=True, alpha=1.0,
                 cbar=False, cbar_orientation='vertical',
                 cbar_kwargs={}, cbar_label=None,
                 interpolation = None, sharex=True, sharey=True,
                 save_name=None, save_kwargs={}):
    """
    Takes either a single im as "im", or a list/array and plots the images in the corresponding shape.
    e.g.
        im = [[im1,im2],
              [im3,im4],
              [im5,im6]]
              generates a 2 column, 3 row figure.

    clim defines the upper and lower limits of the color stretch for the plot.

    If clim is a string, it should contain a comma separating
    two entries. These entries should be one of:
    a) interpretable as a float, in which case they serve as the 
    corresponding entry in the utilized clim, b) they should contain a
    % symbol, in which case they are used as a percentile bound;
    e.g., clim='0, 99.9%' will yield an image with a color
    stretch spanning [0, np.nanpercentile(im, 99.9)], or c) they
    should contain a '*' symbol, separating either of the 
    aforementioned options, in which case they will be multiplied 
    thusly; e.g., clim='0.01*99.9%, 99.9%' would yield a plot with 
    colormapping spanning two decades (i.e., maybe appropriate for
    a logarithmic norm): 
    [0.01*np.nanpercentile(im, 99.9), np.nanpercentile(im, 99.9)].

    If clim is None, clim_perc is used to compute a clim instead. If
    clim_perc contains two values, these are the lower and upper limit
    percentiles. If only a single value, P, is given, a symmetric clim is
    generated spanning plus and minus the P-percentile of the absolute
    value of im (best used with a diverging/symmetric colormap, such as
    'coolwarm'). 
    """   

    if isinstance(clim, str):
        s_clim = [i.strip() for i in clim.split(',')]
        clim = []
        for s in s_clim:
            if s.isdigit():
                clim.append(float(s))
            elif '%' in s:
                if '*' in s:
                    svals = []
                    for si in s.split('*'):
                        if '%' in si:
                            svals.append(np.nanpercentile(im, float(si.replace('%',''))))
                        else:
                            svals.append(float(si))
                    clim.append(np.product(svals))
                else:
                    clim.append(np.nanpercentile(im, float(s.replace('%',''))))
            else:
                raise ValueError(
                    """
                    If clim is a string, it should contain a comma separating
                    two entries. These entries should be one of:
                    a) interpretable as a float, in which case they serve as the 
                    corresponding entry in the utilized clim, b) they should contain a
                    % symbol, in which case they are used as a percentile bound;
                    e.g., clim='0, 99.9%' will yield an image with a color
                    stretch spanning [0, np.nanpercentile(im, 99.9)], or c) they
                    should contain a '*' symbol, separating either of the 
                    aforementioned options, in which case they will be multiplied.
                    """)
            
    elif isNone(clim):
        if np.isscalar(clim_perc) or len(clim_perc) == 1:
            clim = symmetric_clim_percentile(im, clim_perc)
        else:
            clim = np.nanpercentile(np.unique(im), clim_perc)
        
    if isNone(ylims):
        ylims = lims
        
    normalization = norm(vmin=clim[0], vmax=clim[1], **norm_kwargs)

    imshape = np.shape(im)
    if isNone(fig_and_ax):
        if len(imshape) == 2:
            nrows = ncols = 1
        elif len(imshape) == 3:
            nrows, ncols = 1, imshape[0]
        elif len(imshape) == 4:
            nrows, ncols = imshape[0], imshape[1]
        else:
            raise ValueError("Argument 'im' must be a 2, 3, or 4 dimensional array")
        n_ims = nrows * ncols
        if isNone(figsize):
            figsize = np.array([ncols,nrows])*np.asarray(panelsize)
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    else:
        n_ims = 1
        fig, ax = fig_and_ax

    if n_ims == 1:
        ax, im = [ax], [im]
    else:
        im = np.asarray(im).reshape((np.product(imshape[0:-2]), imshape[-2], imshape[-1]))
        ax = np.asarray(ax).flatten()

    for ax_i, im_i in zip(ax, im):
        implot = ax_i.imshow(im_i, origin='lower', cmap=cmap, norm=normalization, extent=extent, alpha=alpha, interpolation=interpolation)

        if not show_ticks:
            ax_i.set(xticks=[], yticks=[])
        ax_i.set(xlim=lims, ylim=ylims)

    if tight_layout:
        fig.tight_layout()
    if cbar:
        cbar = fig.colorbar(implot, ax=ax, orientation=cbar_orientation, **cbar_kwargs)
        cbar.set_label(cbar_label)
    if not isNone(save_name):
        plt.savefig(save_name, bbox_inches='tight', **save_kwargs)
    if show:
        plt.show()
        return None
    if n_ims == 1:
        ax = ax[0]
    return fig, ax


def animate_quick_implot(im_cube, dur=3, titles=None, border_color='k', fig_ax_fns=None, title_pad = 15, title_fontsize = None, **quick_implot_kwargs):

    if isNone(titles): titles = np.arange(len(im_cube)).astype(str)
    else: titles = np.asarray(titles).astype(str)

    fig,ax = quick_implot(im_cube[0], show=False, **quick_implot_kwargs)
    children = ax.get_children()
    im_ind = np.where([type(child) == mpl.image.AxesImage for child in children])[0][0]
    implot = children[im_ind]
    ax.set_title(titles[0], pad=10)
    [ax.spines[key].set_edgecolor(border_color) for key in ax.spines]

    if not isNone(fig_ax_fns):
        for fn in fig_ax_fns:
            fig, ax = fn(fig, ax)
        
    def animate(ind):
        implot.set_data(im_cube[ind])
        ax.set_title(titles[ind], pad=title_pad, fontsize=title_fontsize)
        return implot
    
    fig.tight_layout()
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    anim = animation.FuncAnimation(fig, animate, frames=len(im_cube), interval=dur*1000 / len(im_cube))
    plt.close()
    return anim


def symmetric_clim_percentile(arr, clim_perc=98):
    clim0 = np.nanpercentile(np.abs(np.unique(arr)), clim_perc)
    return np.array([-1, 1])*clim0


def add_iwa_mask(ax, iwa=0.63, cent=(0,0), facecolor='white', edgecolor='k', linestyle='dashed', zorder=1, linewidth=2.0, marker='+', marker_size = 500, marker_color=None, marker_lw=None):
    patch = mpl.patches.Circle(cent, iwa, facecolor=facecolor, edgecolor=edgecolor, ls=linestyle, zorder=zorder, lw=linewidth)
    ax.add_patch(patch)
    if not isNone(marker):
        if isNone(marker_color):
            marker_color = edgecolor
        if isNone(marker_lw):
            marker_lw = linewidth
        ax.scatter(*cent, c=marker_color, marker=marker, s=marker_size, lw=marker_lw, zorder=zorder+0.1)
    return ax


def model_rescale_factor(A, B, sig=None, mask=None):
    """
    Determines the value of scalar c such that:

        chi^2 = sum [ (A-c*B)^2 / sig^2 ]

    is minimized.

    Parameters
    ----------
    A : numpy.ndarray
        Array of measurements

    B : numpy.ndarray
        Array of model values. Shape must match A and B

    sig : numpy.ndarray, optional
        The 1 sigma uncertainty for the measurements of A.

    mask : numpy.ndarray, optional
        A boolean mask with False for entries of A, B, and sig not to be
        utilized, and True for entries that are. Defaults to None.

    Returns
    -------
    c : float
        The scaling factor to multiply the model (B) by to achieve the minimum chi^2
        for measurements (A) having the given uncertainties (sig).
    """

    if np.shape(A) != np.shape(B):
        raise ValueError("A and B must be arrays of the same shape!")

    if not isNone(sig):
        if np.shape(A) != np.shape(sig):
            raise ValueError("A, B, and sig must be arrays of the same shape if sig is specified!")
    else:
        sig = 1
    if isNone(mask):
        c = np.nansum(A * B / (sig ** 2)) / np.nansum((B ** 2) / (sig ** 2))
    elif np.shape(mask) != np.shape(A):
        raise ValueError("If provided, mask must have the same shape as A, B, and sig!")
    else:
        Amsk, Bmsk = A[mask], B[mask]
        if np.ndim(sig) != 0:
            Smsk = sig[mask]
        else:
            Smsk = sig
        c = np.nansum(Amsk * Bmsk / (Smsk ** 2)) / np.nansum((Bmsk ** 2) / (Smsk ** 2))
    return c


def ang_displacement(r, dtheta):
    """
    Returns displacement for a point at radius r rotated by an angle dtheta (deg)
    """
    return 2 * r * np.abs(np.sin(np.deg2rad(dtheta / 2.)))


def nan_median_absolute_deviation(x, axis=None, scaled=True, return_median=False):
    """
    Median absolute deviation, optionally scaled to serve as a consistent estimator
    """
    med = np.nanmedian(x, axis=axis)
    mad = np.nanmedian(np.abs(x - med), axis=axis)
    if scaled:
        mad *= 1.4826
    if return_median:
        return (mad, med)
    return mad


def nan_median_absolute_deviation_gpu(x0, axis=None, scaled=True, return_median=False):
    """
    Median absolute deviation, optionally scaled to serve as a consistent estimator
    """    
    x = cp.asarray(x0)
    
    med_cp = gpu_nanmedian(x, axis=axis)
    free_gpu()
    
    abs_dev = cp.abs(x - med_cp)
    x = free_gpu(x)
    
    mad_cp = gpu_nanmedian(abs_dev, axis=axis)
    abs_dev = free_gpu(abs_dev)

    if scaled: mad_cp *= 1.4826
        
    mad, med = cp.asnumpy(mad_cp), cp.asnumpy(med_cp)
    mad_cp, med_cp = free_gpu(mad_cp, med_cp)
    if return_median:
        return (mad, med)
    return mad


def sigma_clipped_axis_nanmean_general(x, n=2., axis=0, clip_mask=None, return_clip_mask=False, use_gpu=False, fast=False):
    if use_gpu:
        res = sigma_clipped_axis_nanmean_gpu(x, n=n, axis=axis, clip_mask=clip_mask, return_clip_mask=return_clip_mask)
    elif fast:
        res = sigma_clipped_axis_nanmean_fast(x, n=n, axis=axis, clip_mask=clip_mask, return_clip_mask=return_clip_mask)
    else:
        res = sigma_clipped_axis_nanmean(x, n=n, axis=axis, clip_mask=clip_mask, return_clip_mask=return_clip_mask)
    return res


def sigma_clipped_axis_nanmean_fast(x, n=2., axis=0, clip_mask=None, return_clip_mask=False):
    """
    Computes the mean of x along the indicated axis, clipping based on sigma from mean
    """
    if isNone(clip_mask):
        sig, mean = np.nanstd(x, axis=axis), np.nanmean(x, axis=axis)
        clip_mask = np.logical_or(np.abs(x - mean) > n*sig, ~np.isfinite(x))
    xma = np.ma.array(x, mask=clip_mask)
    clipped_mean = np.ma.mean(xma, axis=axis)
    x_out = clipped_mean.data
    x_out[np.all(np.isnan(x), axis=axis)] = np.nan
    if return_clip_mask:
        return x_out, clip_mask
    return x_out


def sigma_clipped_axis_nanmean(x, n=2., axis=0, clip_mask=None, return_clip_mask=False):
    """
    Computes nanmean of x along the indicated axis, clipping values further than n MAD from the median (non-iteratively).
    """
    if isNone(clip_mask):
        sig_mad, median = nan_median_absolute_deviation(x, axis=axis, scaled=True, return_median=True)
        clip_mask = np.logical_or(np.abs(x - median) > n * sig_mad, ~np.isfinite(x))
    xma = np.ma.array(x, mask=clip_mask)
    clipped_mean = np.ma.mean(xma, axis=axis)
    x_out = clipped_mean.data
    x_out[np.all(np.isnan(x), axis=axis)] = np.nan
    if return_clip_mask:
        return x_out, clip_mask
    return x_out


def sigma_clipped_axis_nanmean_gpu(x, n=2., axis=0, clip_mask=None, return_clip_mask=False):
    """
    Computes nanmean of x along the indicated axis, clipping values further than n MAD from the median (non-iteratively).
    """
    if isNone(clip_mask):
        sig_mad, median = nan_median_absolute_deviation_gpu(x, axis=axis, scaled=True, return_median=True)
        clip_mask = np.abs(x - median) > (n * sig_mad)
    clip_mask_cp = cp.asarray(clip_mask)
    x_cp = cp.asarray(x)
    xma = cp.where(clip_mask_cp, cp.nan, x_cp)
    x_cp, clip_mask_cp = free_gpu(x_cp, clip_mask_cp)
    
    x_out_cp = cp.nanmean(xma, axis=axis)
    x_out = cp.asnumpy(x_out_cp)
    xma, x_out_cp = free_gpu(xma, x_out_cp)
    if return_clip_mask:
        return x_out, clip_mask
    return x_out
    

def expand_mask(mask, radius=None, size=None, footprint=None, prop_threshold=1e-6):
    if isNone(footprint):
        if not isNone(radius):
            footprint = Tophat2DKernel(radius, mode='oversample', factor=100).array
        if not isNone(size):
            footprint = np.ones((size, size))
    fn_args, fn_kwargs = [footprint], dict(mode='same')
    im_mask = np.where(mask, np.nan, 0)
    im_out = propagate_nans_in_spatial_operation(im_mask, signal.fftconvolve, fn_args=fn_args, fn_kwargs=fn_kwargs, prop_threshold=prop_threshold)
    mask_out = np.isnan(im_out)
    return mask_out


def ang_size_to_proj_sep(ang_size, distance=None):
    """
    Converts angular separation (any angular unit, e.g. arcsec, degrees, radians) to projected separation (au).
    ang_size and distance can be provided as a float/int (in which case units of arcsec and parsec are assumed
    respectively).
    
    If not specified, units for ang_size and distance are assumed to be arcseconds and parsecs respectively.
    
    Example:
        1) r = ang_size_to_proj_sep(0.25, 156) 
        # Returns the proj separation in astropy units of au for an angular separation of 0.25 arcsec at 156 pc
        
        2) r = ang_size_to_proj_sep(250*u.mas, 508.8*u.lightyear) 
        # Returns roughly the same value as example 1, but with different input units.
        
    Note: returns an astropy unit value. 
          ang_size_to_proj_sep(ang_size, distance).value will give you a float instead.
    """
    if isNone(distance):
        if 'd' in target_defaults:
            distance = target_defaults['d']
        else:
            raise ValueError("Argument 'distance' must either be specified or set in the target_defaults"
            "dictionary (under key 'd')")
    ang_size = ang_size << u.arcsec # If units aren't provided, sets unit to arcsec. Else converts unit to arcsec
    d = distance << u.pc
    return (d * np.tan(ang_size.to('rad'))).to('AU')


def proj_sep_to_ang_size(proj_sep, distance=None):
    """
    Converts projected size (any unit of length) to angular separation (in arcsec)
    
    If not specified, units for proj_sep and distance are assumed to be AU and parsec respectively.
    """
    if isNone(distance):
        if 'd' in target_defaults:
            distance = target_defaults['d']
        else:
            raise ValueError("Argument 'distance' must either be specified or set in the target_defaults"
            "dictionary (under key 'd')")
    r = proj_sep << u.au
    d = distance << u.pc
    r = r << u.pc
    return np.arctan2(r, d).to(u.arcsec)


def ang_size_to_px_size(ang_size, pxscale=0.0162*(u.arcsec/u.pixel)):
    """
    Converts an angular separation (any angular unit) to pixels (based on pixel scale provided).
    
    If not specified, units for ang_size and pxscale are assumed to be arcseconds and arcsec/pixel respectively.
    """
    ang_size = ang_size << u.arcsec
    pxscale = pxscale << u.arcsec / u.pixel
    return ang_size / pxscale


def px_size_to_ang_size(px_size, pxscale=0.0162 * (u.arcsec / u.pixel)):
    """
    Converts a pixel size (in pixels) to an angular separation (in arcsec).

    If not specified, units for px_size and pxscale are assumed to be pixels and arcsec/pixel respectively.
    """
    px_size = px_size << u.pixel
    pxscale = pxscale << u.arcsec / u.pixel
    return px_size * pxscale


def crop_data(data, new_shape, cent=None):
    """
    Crops N>2 dimensional "data" along final two axes with no interpolation, etc.
    """
    ny, nx = data.shape[-2:]
    if isNone(cent):
        cent = (np.array([nx, ny])-1.)/2.
    new_ny, new_nx = new_shape
    x0, y0 = cent
    x1, y1 = max(0, int(np.round(x0-(new_nx-1.)/2.))), max(0, int(np.round(y0-(new_ny-1.)/2.)))
    x2, y2 = x1+new_nx, y1+new_ny
    data_cropped = data[..., y1:y2, x1:x2].copy()
    new_cent = np.array([x0-x1, y0-y1])
    return data_cropped, new_cent


def pad_or_crop_image(im, new_size, cent=None, new_cent=None, cval=np.nan, prop_threshold=1e-6, order=3, mode='constant', prefilter=True):
    new_size = np.asarray(new_size)
    im_size = np.array(im.shape)
    ny, nx = im_size
    
    if isNone(cent):
        cent = (np.array([nx,ny])-1.)/2.
        
    if np.all([new_size == im_size, cent == new_cent]):
        return im.copy()
    
    im_out = propagate_nans_in_spatial_operation(im, pad_or_crop_about_pos,
                                                 fn_args=[cent, new_size],
                                                 fn_kwargs=dict(new_cent=new_cent, cval=cval,
                                                                order=order, mode=mode,
                                                                prefilter=prefilter),
                                                 fn_nan_kwargs=dict(new_cent=new_cent, cval=cval,
                                                                order=order, mode=mode,
                                                                prefilter=False),
                                                 prop_threshold=prop_threshold)
    return im_out


def pad_or_crop_about_pos(im, pos, new_size, new_cent=None, cval=np.nan, order=3, mode='constant', prefilter=True):
    ny, nx = im.shape[-2:]
    ny_new, nx_new = new_size
    if isNone(new_cent):
        new_cent = (np.array([nx_new,ny_new])-1.)/2.
        
    nd = np.ndim(im)
    xg, yg = np.meshgrid(np.arange(nx_new, dtype=np.float64), np.arange(ny_new, dtype=np.float64))
    
    xg -= (new_cent[0]-pos[0])
    yg -= (new_cent[1]-pos[1])

    if nd == 2:
        im_out = ndimage.map_coordinates(im, np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
    else:
        nI = np.product(im.shape[:-2])
        im_reshaped = im.reshape((nI, ny, nx))
        im_out = np.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_out[i] = ndimage.map_coordinates(im_reshaped[i], np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
        im_out = im_out.reshape((*im.shape[:-2], ny, nx))
    return im_out


def median_filter_sequence(im, radius=None, size=None, footprint=None, prop_threshold=1e-6):
    im = np.asarray(im)
    nd = np.ndim(im)
    fn_args = []
    fn_kwargs = dict(size=size, footprint=footprint)
    if not isNone(radius) and isNone(footprint):
        rnx = rny = int(np.ceil(radius)*2+1)
        rc0 = np.array([(rnx-1)/2.,(rnx-1)/2.])
        footprint = dist_to_pt(rc0, nx=rnx, ny=rny) <= radius
        fn_kwargs['footprint'] = footprint
    if nd == 2:
        im_out = propagate_nans_in_spatial_operation(im, ndimage.median_filter, fn_args=fn_args, fn_kwargs=fn_kwargs, prop_threshold=prop_threshold)
    else:
        ny, nx = im.shape[-2:]
        nI = np.product(im.shape[:-2])
        im_reshaped = im.reshape((nI, ny, nx))
        im_out = np.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_out[i] = propagate_nans_in_spatial_operation(im_reshaped[i], ndimage.median_filter, fn_args=fn_args, fn_kwargs=fn_kwargs, prop_threshold=prop_threshold)
        im_out = im_out.reshape(im.shape)
    return im_out


def gaussian_filter_sequence(im, sigma, prop_threshold=1e-6):
    im = np.asarray(im)
    nd = np.ndim(im)
    fn_args = []
    fn_kwargs = dict(sigma=sigma)
    if nd == 2:
        im_out = propagate_nans_in_spatial_operation(im, ndimage.gaussian_filter, fn_args=fn_args, fn_kwargs=fn_kwargs, prop_threshold=prop_threshold)
    else:
        ny, nx = im.shape[-2:]
        nI = np.product(im.shape[:-2])
        im_reshaped = im.reshape((nI, ny, nx))
        im_out = np.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_out[i] = propagate_nans_in_spatial_operation(im_reshaped[i], ndimage.gaussian_filter, fn_args=fn_args, fn_kwargs=fn_kwargs, prop_threshold=prop_threshold)
        im_out = im_out.reshape(im.shape)
    return im_out


def propagate_nans_in_spatial_operation(a, fn, fn_args=None,
                                        fn_kwargs=None,
                                        fn_nan_kwargs=None,
                                        fn_zero_kwargs=None,
                                        prop_threshold=0,
                                        prop_zeros=True):
    """
    This takes an array, a, and and a function that performs some spatial operation on a, fn,
    and attempts to propgate any nans (and optionally: zeros, which are often also non-physical values)
    through the indicated operation. Note: this operation is intentionally liberal with propgating the specified values.
    I.e., for rotation of an image with nans, expect there to be more NaN pixels following the operation. 
    This can be tuned somewhat by increasing the value of prop_threshold (0 <= prop_threshold <= 1)
    
    Example:

    import numpy as np
    from scipy import ndimage
    im = np.random.normal(loc=10, size=(101,101))
    im = ndimage.gaussian_filter(im, sigma=2.5)
    im[68:75, 34:48] = np.nan
    im[11:22, 8:19] = 0.
    angle = 30.0 # angle to rotate image by
    im_rot = propagate_nans_in_spatial_operation(im, ndimage.rotate, fn_args=[angle],
                                                 fn_kwargs=dict(axes=(-2, -1), reshape=False, cval=np.nan, prefilter=False),
                                                 fn_nan_kwargs=dict(axes=(-2, -1), reshape=False, prefilter=False),
                                                 prop_threshold=0, prop_zeros=True)
    """
    if isNone(fn_args): fn_args = []
    if isNone(fn_kwargs): fn_kwargs = {}
    if isNone(fn_nan_kwargs): fn_nan_kwargs = fn_kwargs
    
    nans = np.isnan(a)
    any_nans = np.any(nans)

    if any_nans:
        a_out = fn(np.where(nans, 0., a), *fn_args, **fn_kwargs)
    else: 
        a_out = fn(a, *fn_args, **fn_kwargs)
        
    if prop_zeros:
        zeros = a == 0.
        any_zeros = np.any(zeros)
        # Apply the operation to the boolean map of zeros 
        # >>> replace any locations > prop_threshold with zeros in the output
        if any_zeros:
            if isNone(fn_zero_kwargs):
                fn_zero_kwargs = fn_nan_kwargs
            zeros_out = fn(zeros.astype(float), *fn_args, **fn_zero_kwargs)
            a_out = np.where(zeros_out>prop_threshold, 0., a_out)
    if any_nans:
        nans_out = fn(nans.astype(float), *fn_args, **fn_nan_kwargs)
        a_out = np.where(nans_out>prop_threshold, np.nan, a_out)
    return a_out


def convolve_with_fwhm_tophat(im, fwhm, prop_threshold=1e-6):
    """
    Often referred to as "aperture summation" in literature; replaces each pixel value 
    with the sum of pixels within r=fwhm/2. (weighted based on the fraction of the pixel
    within the aperture).
    """
    nd = np.ndim(im)
    
    psf_kernel = Tophat2DKernel(fwhm / 2., mode='oversample', factor=100).array
    psf_kernel /= np.nanmax(psf_kernel)
    fn_args, fn_kwargs = [psf_kernel], dict(mode='same')
    if nd == 2:
        im_con = propagate_nans_in_spatial_operation(im, signal.fftconvolve, fn_args=fn_args, fn_kwargs=fn_kwargs, prop_threshold=prop_threshold)
    else:
        ny, nx = im.shape[-2:]
        nI = np.product(im.shape[:-2])
        im_reshaped = im.reshape((nI, ny, nx))
        im_con = np.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_con[i] = propagate_nans_in_spatial_operation(im_reshaped[i], signal.fftconvolve, fn_args=fn_args, fn_kwargs=fn_kwargs, prop_threshold=prop_threshold)
        im_con = im_con.reshape(im.shape)
    return im_con


def bin_to_psf_fwhm(im, fwhm):
    """ 
    Resizes the image using scipy.ndimage.zoom such that each pixel is fwhm*fwhm. It seems that the IDL code ensures that an image with its
    center at a pixel will ensure the center still falls on a single pixel after resizing, even if this wouldn't normally be the case
    -- e.g. a 201x201 image to 76x76 image. Shifts product half pixel right to accomodate. We do this here, as it appears to be necessary
    to reproducing the chisq values within error.
    
    In the DPP, this is applied to images having already been convolved with the FWHM-sized tophat function (convolve_with_fwhm_tophat() here)
    and rad profile subtracted (radial_prof_subtract_image)
    
    i.e. 'imcon' to 'imbin'
    """
    im_prepped = deepcopy(im)
    im_prepped[np.isnan(im_prepped)] = 0.0
    
    nx = im.shape[1]
    new_nx = int(round(nx/fwhm))
    offset = 1 - new_nx%2 # e.g. offset is 1 if new_nx is even, 0 if new_nx is odd.

    imbin = ndimage.zoom(im_prepped, (new_nx+offset)/nx, order=1, prefilter=False)
    imbin = imbin[:new_nx, :new_nx]
    return imbin


def model_Fpol(incl, pa, a, b, c, s=1, nx=201, ny=201, pixscale=0.0162, cent=None, nr=50, rmax_mult=10, grid_method="cubic", distance=None):
    from scipy.interpolate import griddata
    def power_law_height(x_power, a_power, b_power, c_power):
        return a_power + b_power * x_power ** c_power
    
    incl_rad = np.deg2rad(incl)
    pos_ang = np.deg2rad(pa)
    pol_max = s

    if isNone(distance):
        distance = target_defaults['d']
        
    if isNone(cent):
        cent = (np.array([nx, ny])-1.)/2.
        
    rmap = dist_to_pt(cent, nx=nx, ny=ny)
    max_radius_in_fov = ang_size_to_proj_sep(px_size_to_ang_size(np.max(rmap)/np.cos(np.deg2rad(incl)), pixscale), distance).value

    r_max = rmax_mult*max_radius_in_fov

    disk_radius = np.logspace(np.log10(1), np.log10(r_max), nr) # midplane radius (au)
    disk_height = power_law_height(disk_radius, a, b, c)
    disk_opening = np.arctan2(disk_height, disk_radius)
    disk_phi = np.deg2rad(np.linspace(0.0, 359.0, 360))  # (rad)

    rg, pg = np.meshgrid(disk_radius, disk_phi)
    hg, _ = np.meshgrid(disk_height, disk_phi)
    og, _ = np.meshgrid(disk_opening, disk_phi)
    rg, pg, hg, og = rg.T.flatten(), pg.T.flatten(), hg.T.flatten(), og.T.flatten()

    x_tmp = rg * np.sin(pg)
    y_tmp = hg * math.sin(incl_rad) - rg * np.cos(pg) * math.cos(incl_rad)
    ang_tmp = 0.5*np.pi + og

    x_im = x_tmp * math.cos(math.pi - pos_ang) - y_tmp * math.sin(math.pi - pos_ang)
    y_im = x_tmp * math.sin(math.pi - pos_ang) + y_tmp * math.cos(math.pi - pos_ang)
    s_im = np.pi - np.arccos(np.sin(ang_tmp) * np.cos(np.pi + pg) * np.sin(incl_rad) + np.cos(ang_tmp) * np.cos(incl_rad))

    image_xy = np.array([x_im, y_im]).T

    xgrid, ygrid = np.meshgrid((np.arange(nx, dtype=np.float32)-cent[0])*pixscale*distance, (np.arange(ny, dtype=np.float32)-cent[1])*pixscale*distance)
    grid = np.array([ygrid.flatten(), xgrid.flatten()]).T
    
    fit_scatter = griddata(image_xy, s_im, grid, method=grid_method).reshape((ny, nx))
    alpha = np.cos(fit_scatter)
    Fpol = -pol_max * (alpha ** 2 - 1.0) / (alpha ** 2 + 1.0)
    return Fpol


def model_rproj(incl, pa, a, b, c, nx=201, ny=201, pixscale=0.0162, cent=None, nr=50, rmax_mult=10, grid_method="cubic", distance=None):
    from scipy.interpolate import griddata
    def power_law_height(x_power, a_power, b_power, c_power):
        return a_power + b_power * x_power ** c_power
    
    incl_rad = np.deg2rad(incl)
    pos_ang = np.deg2rad(pa)

    if isNone(distance):
        distance = target_defaults['d']
        
    if isNone(cent):
        cent = (np.array([nx, ny])-1.)/2.
        
    rmap = dist_to_pt(cent, nx=nx, ny=ny)
    max_radius_in_fov = ang_size_to_proj_sep(px_size_to_ang_size(np.max(rmap)/np.cos(np.deg2rad(incl)), pixscale), distance).value

    r_max = rmax_mult*max_radius_in_fov

    disk_radius = np.logspace(np.log10(1), np.log10(r_max), nr) # midplane radius (au)
    disk_height = power_law_height(disk_radius, a, b, c)
    disk_opening = np.arctan2(disk_height, disk_radius)
    disk_phi = np.deg2rad(np.linspace(0.0, 359.0, 360))  # (rad)

    rg, pg = np.meshgrid(disk_radius, disk_phi)
    hg, _ = np.meshgrid(disk_height, disk_phi)
    og, _ = np.meshgrid(disk_opening, disk_phi)
    rg, pg, hg, og = rg.T.flatten(), pg.T.flatten(), hg.T.flatten(), og.T.flatten()

    x_tmp = rg * np.sin(pg)
    y_tmp = hg * math.sin(incl_rad) - rg * np.cos(pg) * math.cos(incl_rad)
    ang_tmp = 0.5*np.pi + og

    x_im = x_tmp * math.cos(math.pi - pos_ang) - y_tmp * math.sin(math.pi - pos_ang)
    y_im = x_tmp * math.sin(math.pi - pos_ang) + y_tmp * math.cos(math.pi - pos_ang)
    s_im = np.pi - np.arccos(np.sin(ang_tmp) * np.cos(np.pi + pg) * np.sin(incl_rad) + np.cos(ang_tmp) * np.cos(incl_rad))

    image_xy = np.array([x_im, y_im]).T

    grid = np.array([i.flatten()*pixscale*distance for i in np.meshgrid(np.arange(nx, dtype=np.float32)-cent[0],
                                                                        np.arange(ny, dtype=np.float32)-cent[1])]).T
    r_im = np.hypot(x_im, y_im)
    fit_rproj = griddata(image_xy, r_im, grid, method=grid_method).reshape((ny, nx))
    return fit_rproj


def imcube_to_adi_hcube(imcube, parangs, ncores=-2, use_gpu=False, cent=None, cval0=0.):
    hcube = rotate_hypercube(np.tile(imcube[np.newaxis], [len(parangs),1,1,1]), parangs, cent=cent, cval0=cval0, ncores=ncores, use_gpu=use_gpu)
    return hcube


def build_sequence_hcube(filepaths, parang_key='parang', 
                         ncores=-2, return_headers=False, 
                         print_approx_ram_usage=False, only_hcube=False,
                         memmap=False, ext=1):
    """
    Load a sequence of images or IFS image cubes into a 4 dimensional "hypercube": (angle/time, wavelength, y, x). 
    If the images are 2D, the wavelength axis will still be present, but simply length 1 instead.
    
    'parang_key' should be a string indicating the header key under which the parallactic angle of each image is stored.
    """
    imlist, headers = zip(
        *Parallel(n_jobs=ncores, prefer="threads")(delayed(fits.getdata)(f, header=True, memmap=memmap, ext=ext)
                                                   for f in filepaths))
    hcube = np.array(imlist)
    if hcube.ndim == 3:
        hcube = hcube[:, np.newaxis]
    if print_approx_ram_usage:
        import sys
        print("Storing the image sequence in RAM requires ~{0:0.2f} GB".format(sys.getsizeof(hcube) / 1e9))
    if only_hcube:
        return hcube
    parangs = np.asarray([h[parang_key] for h in headers])
    if return_headers:
        return hcube, parangs, headers
    return hcube, parangs


def cube_combine(imcube, exclude_tellurics=True):
    if exclude_tellurics:
        return np.mean([*imcube[0:5], *imcube[7:14], *imcube[15:21]], axis=0)
    return np.mean(imcube, axis=0)


def check_dir(path):
    """
    Checks if a directory exists and creates it if it doesn't exist already.
    """
    import sys
    import os
    if not os.path.isdir(path):
        os.system('mkdir {}'.format(path))
    return None


try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    from cupyx.scipy import signal as cp_signal
    use_gpu = True
    gpu = cp.cuda.Device(0)
    print("CuPy succesfully imported. Using GPU where applicable. "
           "Set use_gpu=False to override this functionality.")
except ModuleNotFoundError:
    use_gpu = False
    print("Could not import CuPy. "
          "Setting: use_gpu=False (i.e., using CPU operations).")