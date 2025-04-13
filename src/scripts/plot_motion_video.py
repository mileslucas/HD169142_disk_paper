import logging

import cv2
import logging_config  # noqa: F401
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import paths
import proplot as pro
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import simple_norm
from matplotlib import patches
from target_info import target_info
from utils_ephemerides import keplerian_warp2d
from utils_indexing import frame_radii
from utils_organization import folders, pxscales, time_from_folder
from utils_plots import setup_rc

logger = logging.getLogger(__file__)

GIF_FPS = 12
YEAR_PER_SEC = 1.5


def inner_ring_mask(frame, radii):
    rin_au = 15
    rout_au = 35
    rad_mask = (radii >= rin_au) & (radii <= rout_au)
    return np.where(rad_mask, frame, np.nan)


def linear_interpolation(t, t0, t1, frame0, frame1):
    ti = (t - t0) / (t1 - t0)
    # allow broadcasting for speed
    frame = (1 - ti[:, None, None]) * frame0 + ti[:, None, None] * frame1
    return frame


def motion_interpolation(t, t0, t1, frame0, frame1, radii_au):
    frames = []
    for ti in t:
        tfrac = (ti - t0) / (t1 - t0)
        time_now = Time(ti, format="mjd")
        frame_b = keplerian_warp2d(frame0, radii_au, Time(t0, format="mjd"), time_now)
        frame_f = keplerian_warp2d(frame1, radii_au, Time(t1, format="mjd"), time_now)
        frame = (1 - tfrac) * frame_b + tfrac * frame_f
        frames.append(frame)

    # allow broadcasting for speed
    return np.array(frames)


def get_frames():
    frames = {}
    r2_maps = {}
    for folder in folders[:-1]:
        filename = (
            paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Qphi_deprojected.fits"
        )
        data = fits.getdata(filename, memmap=False)
        rs_px = frame_radii(data)
        rs_au = rs_px * pxscales[folder] * target_info.dist_pc
        r2_map = rs_au**2
        r2_maps[folder] = r2_map
        frame_r2 = data * r2_map
        frames[folder] = frame_r2

    return frames, r2_maps


def regrid_frames(frames, r2_maps):
    # determine the finest spacing
    min_pxscale = min(pxscales.values())
    spacing_au = min_pxscale * target_info.dist_pc
    # determine maximum extent from smallest FOV
    max_rad = np.sqrt(min(np.max(m) for m in r2_maps.values()))
    grid_vec = np.arange(-max_rad, max_rad, spacing_au)

    grid_xs_au, grid_ys_au = np.meshgrid(grid_vec, grid_vec)

    regridded = {}
    for folder, frame in frames.items():
        cy, cx = np.array(frame.shape) / 2 - 0.5
        grid_ys = grid_ys_au / (pxscales[folder] * target_info.dist_pc) + cy
        grid_xs = grid_xs_au / (pxscales[folder] * target_info.dist_pc) + cx
        regridded_frame = cv2.remap(
            frame.astype("f4"),
            grid_xs.astype("f4"),
            grid_ys.astype("f4"),
            interpolation=cv2.INTER_LANCZOS4,
        )
        regridded[folder] = regridded_frame
    return regridded, min_pxscale


def normalize_frames(frames, r2_maps):
    output = {}
    for folder in frames:
        frame = frames[folder]
        r2_map = r2_maps[folder]
        _masked = inner_ring_mask(frame, np.sqrt(r2_map))
        vmax = np.nanmax(_masked)
        norm = simple_norm(frame, vmin=0, vmax=vmax, stretch="sinh", sinh_a=0.5)
        # norm = simple_norm(frame, vmin=0, vmax=vmax)
        output[folder] = norm(frame, clip=True)
    return output


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


def plot_frames(frames, timestamps, pxscale):
    # Create figure
    width = 3.31314
    fig, ax = pro.subplots(width=f"{width}in")

    side_length = frames[0].shape[-1] * pxscale / 2
    ext = (side_length, -side_length, -side_length, side_length)

    image = ax.imshow(frames[0], cmap="bone", extent=ext, vmin=0, vmax=1)
    label = ax.text(
        0.03,
        0.97,
        _str_from_timestamp(timestamps[0]),
        fontsize=9,
        transform="axes",
        c="w",
        fontweight="bold",
        ha="left",
        va="top",
    )

    # star position
    ax.scatter(0, 0, marker="+", lw=1, markersize=50, c="white")
    # scale bar
    bar_width_au = 15
    bar_width_arc = bar_width_au / target_info.dist_pc
    bar_width_height = bar_width_arc / 20
    rect = patches.Rectangle(
        [0.3, -0.32 - bar_width_height / 2], -bar_width_arc, bar_width_height, color="white"
    )
    ax.add_patch(rect)

    ax.text(
        0.3 - bar_width_arc / 2,
        -0.32 + bar_width_arc / 5,
        f"{bar_width_au:.0f} au",
        c="white",
        ha="center",
        fontsize=9,
    )

    ax.format(xlim=(0.35, -0.35), ylim=(-0.35, 0.35), xlocator="none", ylocator="none")

    def update(idx):
        image.set_data(frames[idx])
        label.set_text(_str_from_timestamp(timestamps[idx]))
        return image, label

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 // GIF_FPS,
        blit=False,
        repeat=True,
        save_count=len(frames),
    )

    # Save video (requires ffmpeg or mencoder)
    plt.show()
    ani.save(
        paths.figures / "HD169142_2012-2024_keplerian.gif",
        writer=animation.ImageMagickFileWriter(fps=GIF_FPS),
        # writer=animation.FFMpegWriter(fps=GIF_FPS, extra_args=['-vcodec', 'libx264']),
        progress_callback=lambda i, n: print(f"Saving frame {i} of {n}"),
    )
    return ani


if __name__ == "__main__":
    setup_rc()
    pro.rc["animation.convert_path"] = "/Users/mileslucas/software/convert"
    logger.info("Loading frames")
    frames, r2_maps = get_frames()

    logger.info("Normalizing frames")
    frames_norm = normalize_frames(frames, r2_maps)
    logger.info("Regridding frames")
    frames_regrid, pxscale = regrid_frames(frames_norm, r2_maps)

    logger.info("Sorting frames")
    keys = sorted(frames_regrid.keys())
    timestamps = [time_from_folder(f).mjd for f in keys]
    frames = [frames_regrid[k] for k in keys]

    # fits.writeto("tmp.fits", np.array(frames), overwrite=True)

    logger.info("Interpolating frames")
    rad_au = frame_radii(frames_regrid[folders[0]]) * pxscale * target_info.dist_pc
    frames_itp, times = interpolate_frames(timestamps, frames, rad_au)

    logger.info("Plotting frames")
    plot_frames(frames_itp, times, pxscale)
