import proplot as pro
import numpy as np
import paths
from astropy.io import fits
from skimage.transform import warp_polar
from astropy.convolution import convolve, kernels
import tqdm
from astropy.visualization import simple_norm
from target_info import target_info
from astropy import time

def cross_correlate(curve1, time1: str, curve2, time2: str):
    time_delta = time_from_folder(time1) - time_from_folder(time2)
    time_delta_yr = time_delta.jd / 365.25 # days to years

    fft1 = np.fft.fft(curve1)
    fft2 = np.fft.fft(curve2)

    R = fft1 * fft2.conj()
    r = np.fft.ifft(R)
    cross_corr = np.real(np.fft.fftshift(r))
    # cross_corr /= cross_corr.max()
    # lags_inds = signal.correlation_lags(len(curve1), len(curve2))

    lags_inds = np.arange(-len(curve1)//2, len(curve2)//2)
    # lags_inds = 1 / np.fft.fftshift(np.fft.fftfreq(len(curve1)))
    # bin_width in azimuthal profile is 5 deg
    lags_deg_per_yr = lags_inds * 5 / time_delta_yr
    return lags_deg_per_yr, cross_corr



def solve_kepler_for_period(separation):
    G = 39.476926408897626 # au^3 / Msun / yr^2
    M = target_info.stellar_mass
    T = np.sqrt(separation**3 * 4 * np.pi**2 / (G * M))
    angular_velocity = 360 / T # deg / yr
    return angular_velocity


def polar_roll_frame(polar_frame, radii_au, time: time.Time, t0: time.Time):
    delta_t_yr = (time - t0).jd / 365.25
    angular_velocity = solve_kepler_for_period(radii_au)
    total_motion = angular_velocity * delta_t_yr
    total_motion_int = np.round(total_motion / 5).astype(int)
    # print(delta_t_yr, total_motion_int)
    # total_motion_int = 
    output_frame = polar_frame.copy()
    for i in range(output_frame.shape[0]):
        output_frame[i] = np.roll(polar_frame[i], (total_motion_int[i], 0))
    return output_frame


def time_from_folder(foldername: str) -> time.Time:
    date_raw = foldername.split("_")[0]
    ymd = {
        "year": int(date_raw[:4]),
        "month": int(date_raw[4:6]),
        "day": int(date_raw[6:])
    }
    return time.Time(ymd, format="ymdhms")


def label_from_folder(foldername):
    tokens = foldername.split("_")
    date = f"{tokens[0][:4]}/{tokens[0][4:6]}/{tokens[0][6:]}"
    return f"{date} {tokens[1]}"


if __name__ == "__main__": 
    pro.rc["image.origin"] = "lower"
    pro.rc["image.cmap"] = "bone"
    pro.rc["axes.grid"] = False
    pro.rc["axes.facecolor"] = "k"
    pro.rc["font.size"] = 8
    pro.rc["title.size"] = 9

    folders = [
        "20120726_NACO",
        "20140425_GPI",
        "20150503_IRDIS",
        "20150710_ZIMPOL",
        "20180715_ZIMPOL",
        "20210906_IRDIS",
        "20230707_VAMPIRES",
        "20240729_VAMPIRES",
    ]
    timestamps = list(map(time_from_folder, folders))
    iwas = {
        "20230707_VAMPIRES": 105,
        "20240727_VAMPIRES": 59,
        "20240728_VAMPIRES": 59,
        "20240729_VAMPIRES": 59
    }

    pxscales = {
        "20120726_NACO": 27e-3,
        "20140425_GPI": 14.14e-3,
        "20150503_IRDIS": 12.25e-3,
        "20150710_ZIMPOL": 3.6e-3,
        "20180715_ZIMPOL": 3.6e-3,
        "20230707_VAMPIRES": 5.9e-3,
        "20210906_IRDIS": 12.25e-3,
        "20240729_VAMPIRES": 5.9e-3,
    }

    ## Plot and save
    fig, axes = pro.subplots(
        nrows=8, width="3.33in", refheight="0.8in", hspace=0.5
    )

    def format_date(date):
        return f"{date[:4]}/{date[4:6]}"

    for i, folder in enumerate(tqdm.tqdm(folders)):

    # load data
        with fits.open(
            paths.data
            / folder
            / f"{folder}_HD169142_Qphi_polar.fits"
        ) as hdul:
            polar_cube = hdul[0].data


        rin = np.floor(15 / target_info.dist_pc / pxscales[folder]).astype(int)
        rout = np.ceil(35 / target_info.dist_pc / pxscales[folder]).astype(int)

        rs = np.arange(polar_cube.shape[0])

        mask = (rs >= rin) & (rs <= rout)
        ext = (0, 360, rin * pxscales[folder] * target_info.dist_pc, rout * pxscales[folder] * target_info.dist_pc)

        polar_cube_rolled = polar_roll_frame(polar_cube[mask, :], rs[mask] * target_info.dist_pc * pxscales[folder], timestamps[i], timestamps[4])

        # PDI images
        norm = simple_norm(polar_cube_rolled, vmin=0, stretch="sinh", sinh_a=0.5)
        im = axes[i].imshow(polar_cube_rolled, extent=ext, norm=norm, vmin=norm.vmin, vmax=norm.vmax, cmap=pro.rc["cmap"])
         # axes[0].colorbar(im)
        labels = label_from_folder(folder).split()
        axes[i].text(
            0.01, 0.95, labels[0], transform="axes", c="white", ha="left", va="top", fontsize=8, fontweight="bold"
        )
        axes[i].text(
            0.99, 0.95, labels[1], transform="axes", c="white", ha="right", va="top", fontsize=8, fontweight="bold"
        )

    for ax in axes:
        norm_pa = np.mod(target_info.pos_angle - 90, 360)
        ax.axvline(norm_pa, lw=1, c="0.8")
        ax.axvline(norm_pa - 180, lw=1, c="0.8")

    ## sup title
    axes.format(
        aspect="auto",
        xlabel="Angle E of N (°)",
        ylabel="Separation (au)",
        xlocator=90,
    ) 

    # axes[:-1].format(xtickloc="none")

    fig.savefig(
        paths.figures / "HD169142_polar_collapsed_inner_rolled.pdf", bbox_inches="tight", dpi=300
    )


