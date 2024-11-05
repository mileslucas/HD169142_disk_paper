from scipy.optimize import minimize_scalar
from utils_indexing import frame_angles
import numpy as np
import logging
import logging_config

logger = logging.getLogger(__file__)


def azimuthal_stokes(Q, U, phi=0):
    angles = frame_angles(Q, conv="astro")
    cos2th = np.cos(2 * (angles + phi))
    sin2th = np.sin(2 * (angles + phi))
    Qphi = -Q * cos2th - U * sin2th
    Uphi = Q * sin2th - U * cos2th
    return Qphi, Uphi


def opt_func(phi, stokes_frame):
    _, Uphi = azimuthal_stokes(stokes_frame[2], stokes_frame[3], phi)
    return np.nanmean(Uphi**2)


def optimize_Uphi(stokes_frames):
    res = minimize_scalar(
        opt_func, args=(stokes_frames,), bounds=(-np.pi / 2, np.pi / 2)
    )
    return res.x


def rotate_stokes(stokes_cube, theta):
    out = stokes_cube.copy()
    sin2ts = np.sin(2 * theta)
    cos2ts = np.cos(2 * theta)
    out[2] = stokes_cube[2] * cos2ts - stokes_cube[3] * sin2ts
    out[3] = stokes_cube[2] * sin2ts + stokes_cube[3] * cos2ts
    return out


def optimize_Uphi_cube(stokes_cube, name="HD169142"):
    output_cube = stokes_cube.copy()
    opt_phis = np.zeros(stokes_cube.shape[0])
    for i in range(stokes_cube.shape[0]):
        stokes_images = stokes_cube[i]
        opt_phis[i] = optimize_Uphi(stokes_images)
        output_cube[i] = rotate_stokes(stokes_images, opt_phis[i])
        Qphi, Uphi = azimuthal_stokes(output_cube[i, 2], output_cube[i, 3])
        output_cube[i, 4] = Qphi
        output_cube[i, 5] = Uphi
    offset_str = ", ".join(f"{np.rad2deg(phi):.01f}°" for phi in opt_phis)
    logger.info(f"{name} phi offsets: {offset_str}")
    return output_cube
