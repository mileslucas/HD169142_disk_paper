from astropy import time
import numpy as np
from target_info import target_info
import cv2

_DEG_PER_PIXEL = 5
_DISK_DIR = "CW"

def blob_c_position(timestamp: time.Time):
    a = 23.1 # au
    w = -4.48 # deg / yr
    t0 = time.Time(58288.19 + 2400000, format="jd")
    theta0 = 299.7

    dt_yr = (timestamp - t0).jd / 365.25
    theta = np.mod(theta0 + dt_yr * w, 360)
    return a, theta


def blob_d_position(timestamp: time.Time):
    a = 36.4 # au
    w = -2.08 # deg / yr
    t0 = time.Time(58288.19 + 2400000, format="jd")
    theta0 = 34.9

    dt_yr = (timestamp - t0).jd / 365.25
    theta = np.mod(theta0 + dt_yr * w, 360)
    return a, theta


def calculate_keplerian_angular_velocity(separation):
    # assumes separation is in au
    G = 39.476926408897626 # au^3 / Msun / yr^2
    M = target_info.stellar_mass # Msun
    omega = np.sqrt(G * M / separation**3) # rad / yr
    angular_velocity = np.rad2deg(omega) # deg / yr
    return angular_velocity


def keplerian_warp(polar_frame, radii_au, time: time.Time, ref_time: time.Time):
    assert len(radii_au) == polar_frame.shape[0]
    delta_t_yr = (time - ref_time).jd / 365.25
    angular_velocity = calculate_keplerian_angular_velocity(radii_au) # deg / yr
    if _DISK_DIR == "CW":
        angular_velocity *= -1
    total_angular_motion = angular_velocity * delta_t_yr # deg
    total_angular_motion_px = total_angular_motion / _DEG_PER_PIXEL # px
        
    r_px, theta_px = np.indices(polar_frame.shape)
    theta_new = np.mod(theta_px + total_angular_motion_px[:, None], polar_frame.shape[1])

    warped_frame = cv2.remap(polar_frame.astype('f4'), theta_new.astype('f4'), r_px.astype('f4'), interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_WRAP)
    return warped_frame

def polar_roll_frame(polar_frame, radii_au, time: time.Time, t0: time.Time):
    delta_t_yr = (time - t0).jd / 365.25
    angular_velocity = calculate_keplerian_angular_velocity(radii_au)
    total_motion = angular_velocity * delta_t_yr
    total_motion_int = np.round(total_motion / 5).astype(int)
    # print(delta_t_yr, total_motion_int)
    # total_motion_int = 
    output_frame = polar_frame.copy()
    for i in range(output_frame.shape[0]):
        output_frame[i] = np.roll(polar_frame[i], (total_motion_int[i], 0))
    return output_frame
