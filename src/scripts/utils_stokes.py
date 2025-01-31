from scipy.optimize import minimize
from utils_indexing import frame_angles, frame_radii
import numpy as np
import logging

logger = logging.getLogger(__file__)


def azimuthal_stokes(Q, U, phi=0):
    angles = frame_angles(Q, conv="astro")
    cos2th = np.cos(2 * (angles + phi))
    sin2th = np.sin(2 * (angles + phi))
    Qphi = -Q * cos2th - U * sin2th
    Uphi = Q * sin2th - U * cos2th
    return Qphi, Uphi


def opt_func(X, stokes_frame):
    phi = X
    Q = stokes_frame[2]
    U = stokes_frame[3]

    _, Uphi = azimuthal_stokes(Q, U, phi)
    return np.nansum(Uphi**2)


def optimize_Uphi(stokes_frames):
    X0 = [0,]
    res = minimize(
        opt_func, X0, args=(stokes_frames,), method="Nelder-Mead"
    )
    return res.x


def rotate_stokes(stokes_cube, theta):
    out = stokes_cube.copy()
    sin2ts = np.sin(2 * theta)
    cos2ts = np.cos(2 * theta)
    out[2] = stokes_cube[2] * cos2ts - stokes_cube[3] * sin2ts
    out[3] = stokes_cube[2] * sin2ts + stokes_cube[3] * cos2ts
    # recalculate
    out[4], out[5] = azimuthal_stokes(out[2], out[3])
    out[6] = np.hypot(out[2], out[3])
    out[7] = 0.5 * np.arctan2(out[3], out[2])
    return out


def measure_unres_pol_coeffs(stokes_cube):
    radii = frame_radii(stokes_cube)
    # two annuli
    # inside inner cavity
    ann_mask = (radii > 20 ) & (radii < 22)
    # one in the known cavity
    ann_mask |= (radii > 42) & (radii < 70)
    # another outside outer ring
    ann_mask |= (radii > 160)

    # Q signal
    Qstar = np.nansum(stokes_cube[2] * ann_mask)
    IQstar = np.nansum(stokes_cube[0] * ann_mask)
    pQ = Qstar / IQstar
    # U signal
    Ustar = np.nansum(stokes_cube[3] * ann_mask)
    IUstar = np.nansum(stokes_cube[1] * ann_mask)
    pU = Ustar / IUstar
    return pQ, pU

def remove_unres_pol(stokes_cube, pQ, pU):
    out = stokes_cube.copy()
    out[2] -= pQ * out[0]
    out[3] -= pU * out[1]
    # recalculate
    out[4], out[5] = azimuthal_stokes(out[2], out[3])
    out[6] = np.hypot(out[2], out[3])
    out[7] = 0.5 * np.arctan2(out[3], out[2])
    return out


def optimize_Uphi_cube(stokes_cube, mask=None, name="HD169142"):
    output_cube = stokes_cube.copy()
    if mask is not None:
        stokes_data = np.where(mask, stokes_cube, 0)
    else:
        stokes_data = stokes_cube

    opt_pQs = np.zeros(stokes_cube.shape[0])
    opt_pUs = np.zeros(stokes_cube.shape[0])
    opt_phis = np.zeros(stokes_cube.shape[0])
    # for i in range(stokes_cube.shape[0]):
    #     opt_pQs[i], opt_pUs[i] = measure_unres_pol_coeffs(stokes_data[i])
    for i in range(stokes_cube.shape[0]):
        # output_cube[i] = remove_unres_pol(stokes_data[i], poly_Q(wl[i]), poly_U(wl[i]))
        opt_phis[i] = optimize_Uphi(stokes_cube[i])
        # opt_pQs[i], opt_pUs[i], opt_phis[i] = X
        # output_cube[i][2] -= opt_pQs[i] * output_cube[i][0]
        # output_cube[i][3] -= opt_pUs[i] * output_cube[i][1]
        output_cube[i] = rotate_stokes(output_cube[i], -opt_phis[i])

        output_cube[i] = rotate_stokes(output_cube[i], -opt_phis[i])
    offset_str = ", ".join(f"{np.rad2deg(phi):.01f}°" for phi in opt_phis)
    print(f"{name} phi offsets: {offset_str}")

    stokes_frames = np.nansum(output_cube, axis=0)
    pQ, pU = measure_unres_pol_coeffs(stokes_frames)
    output_frame = remove_unres_pol(stokes_frames, pQ, pU)

    # wl = [614, 670, 721, 761]
    # test_wl = np.linspace(600, 800, 500)
    # poly_Q = np.polynomial.Polynomial.fit(wl, opt_pQs, deg=2).convert()
    # poly_U = np.polynomial.Polynomial.fit(wl, opt_pUs, deg=2).convert()
    # import proplot as pro
    # fig, axes = pro.subplots()
    # axes[0].scatter(wl, opt_pQs, label="Q", c="C0")
    # axes[0].plot(test_wl, poly_Q(test_wl), c="C0")
    # axes[0].scatter(wl, opt_pUs, label="U", c="C1")
    # axes[0].plot(test_wl, poly_U(test_wl), c="C1")
    # axes[0].legend()
    # pro.show(block=True)
    # for i in range(stokes_cube.shape[0]):
    #     output_cube[i] = remove_unres_pol(stokes_data[i], poly_Q(wl[i]), poly_U(wl[i]))
    #     opt_phis[i] = optimize_Uphi(output_cube[i])
    #     # opt_pQs[i], opt_pUs[i], opt_phis[i] = X
    #     # output_cube[i][2] -= opt_pQs[i] * output_cube[i][0]
    #     # output_cube[i][3] -= opt_pUs[i] * output_cube[i][1]
    return output_frame
