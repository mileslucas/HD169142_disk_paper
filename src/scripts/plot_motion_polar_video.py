import paths
import proplot as pro
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import simple_norm
from utils_plots import setup_rc
from utils_organization import folders, time_from_folder, pxscales
from utils_indexing import frame_radii
from astropy.io import fits
from target_info import target_info
from scipy import interpolate
import logging_config
import logging
from matplotlib import patches
from utils_ephemerides import keplerian_warp
from astropy.time import Time
import cv2

logger = logging.getLogger(__file__)

GIF_FPS = 12
YEAR_PER_SEC = 1.5


def motion_interpolation(t, t0, t1, frame0, frame1, radii_au):

    frames = []
    for ti in t:
        tfrac = (ti - t0) / (t1 - t0)
        time_now = Time(ti, format="mjd")
        frame_b = keplerian_warp(frame0, radii_au, Time(t0, format="mjd"), time_now)
        frame_f = keplerian_warp(frame1, radii_au, Time(t1, format="mjd"), time_now)
        frame = (1 - tfrac) * frame_b + tfrac * frame_f
        frames.append(frame)

    # allow broadcasting for speed
    return np.array(frames)


def get_frames():
    frames = {}
    rad_aus = {}
    for folder in folders[:-1]:
        filename = paths.data / folder / f"{folder}_HD169142_Qphi_polar.fits"
        polar_frame = fits.getdata(filename, memmap=False)

        rin = np.floor(15 / target_info.dist_pc / pxscales[folder]).astype(int)
        rout = np.ceil(35 / target_info.dist_pc / pxscales[folder]).astype(int)
        
        rs = np.arange(polar_frame.shape[0])
        mask = (rs >= rin) & (rs <= rout)
        rad_aus[folder] = rs[mask] * target_info.dist_pc * pxscales[folder]

        # PDI images
        polar_frame_masked = polar_frame[mask, :]
        frames[folder] = polar_frame_masked

    return frames, rad_aus


def normalize_frames(frames):
    output = {}
    for folder in frames.keys():
        frame = frames[folder]
        norm = simple_norm(frame, vmin=0, stretch="sinh", sinh_a=0.5)
        output[folder] = norm(frame, clip=True)
    return output

def regrid_frames(frames, rad_aus):
    # determine the finest spacing
    min_pxscale = min(pxscales.values())
    spacing_au = min_pxscale * target_info.dist_pc
    # determine maximum extent from smallest FOV
    common_rs = np.arange(15, 35 + spacing_au/2, spacing_au)
    common_thetas = np.arange(0, 360 // 5)

    grid_ts, grid_rs = np.meshgrid(common_thetas, common_rs)

    regridded = {}
    for folder, frame in frames.items():
        grid_rs_norm = (grid_rs - rad_aus[folder].min()) / (pxscales[folder] * target_info.dist_pc)
        regridded_frame = cv2.remap(frame.astype("f4"), grid_ts.astype("f4"), grid_rs_norm.astype("f4"), cv2.INTER_LANCZOS4)

        regridded[folder] = regridded_frame
    return regridded, common_rs

def interpolate_frames(timestamps, frames, rad_au):
    # timing, we 
    total_year = (timestamps[-1] - timestamps[0]) / 365.25
    total_frames = int(total_year / YEAR_PER_SEC * GIF_FPS)
    times = np.linspace(timestamps[0], timestamps[-1], total_frames)

    output = []
    for i in range(len(timestamps) - 1):
        t0 = timestamps[i]
        t1 = timestamps[i + 1]
        ts = times[(times >= t0) & (times < t1)]
        _frames = motion_interpolation(ts, t0, t1, frames[i], frames[i + 1], rad_au)
        # _frames = linear_interpolation(ts, t0, t1, frames[i], frames[i + 1])
        output.extend(_frames)

    return np.array(output), times


def _str_from_timestamp(timestamp):
    time = Time(timestamp, format="mjd")
    return time.strftime("%Y/%m")

def plot_frames(frames, timestamps, rad_au):
    # Create figure
    width = 3.31314
    height = width / 2.5
    fig, ax = pro.subplots(width=f"{width}in", height=f"{height}in")

    ext = (0, 360, rad_au.min(), rad_au.max())

    image = ax.imshow(frames[0], cmap="bone", extent=ext)

    label = ax.text(
       0.01, 0.95, _str_from_timestamp(timestamps[0]), transform="axes", c="white", ha="left", va="top",  fontweight="bold"
    )

    ax.format(
        aspect="auto",
        xlabel="Angle E of N (°)",
        ylabel="Separation (au)",
        xlocator=90,
    )

    def update(idx):
        image.set_data(frames[idx])
        label.set_text(_str_from_timestamp(timestamps[idx]))
        return image,label

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 // GIF_FPS, blit=False, repeat=True, save_count=len(frames))

    # Save video (requires ffmpeg or mencoder)
    plt.show()
    ani.save(
        paths.figures / "HD169142_2012-2024_polar_keplerian.gif", 
        writer=animation.ImageMagickFileWriter(fps=GIF_FPS),
        # writer=animation.FFMpegWriter(fps=GIF_FPS, extra_args=['-vcodec', 'libx264']),
        progress_callback=lambda i, n: print(f'Saving frame {i} of {n}')
    )
    return ani

if __name__ == "__main__":
    setup_rc()
    pro.rc["grid"] = False
    pro.rc["animation.convert_path"] = "/Users/mileslucas/software/convert"

    logger.info("Loading frames")
    frames, rad_aus = get_frames()

    logger.info("Normalizing frames")
    frames_norm = normalize_frames(frames)
    logger.info("Regridding frames")
    frames_regrid, common_rad_au = regrid_frames(frames_norm, rad_aus)

    logger.info("Sorting frames")
    keys = sorted(frames_regrid.keys())
    timestamps = [time_from_folder(f).mjd for f in keys]
    frames = [frames_regrid[k] for k in keys]

    # fits.writeto("tmp.fits", np.array(frames), overwrite=True)

    logger.info("Interpolating frames")
    frames_itp, times = interpolate_frames(timestamps, frames, common_rad_au)


    logger.info("Plotting frames")
    plot_frames(frames_itp, times, common_rad_au)
   