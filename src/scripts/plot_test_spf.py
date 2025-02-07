import paths
import proplot as pro
import numpy as np
from target_info import target_info
from utils_plots import setup_rc

def rayleigh(scat_angle, pol_max=1):
    alpha = np.cos(scat_angle)
    return pol_max * (1 - alpha**2) / (alpha**2 + 1)


def hg(scat_angle, pol_max=1, g=0):
    if g <= -1 or g >= 1:
        msg = f"g parameter must be between (-1, 1); got {g=}"
        raise ValueError(msg)
    ray = rayleigh(scat_angle, pol_max=pol_max)
    hg_fac = (1 - g**2) / np.sqrt((1 + g**2 - 2 * g * np.cos(scat_angle)) ** 3)
    return hg_fac * ray


if __name__ == "__main__":
    setup_rc()

    width = 3.31314
    aspect_ratio = 1 / 1.6
    height = width * aspect_ratio
    fig, axes = pro.subplots(width=f"{width}in", height=f"{height}in")

    test_angles = np.linspace(0, 180, 1000)

    min_angle = 90 - target_info.inclination
    max_angle = 90 + target_info.inclination
    mask = (min_angle <= test_angles) & (test_angles <= max_angle)

    hg_0 = hg(np.deg2rad(test_angles), g=0)

    axes[0].plot(test_angles, hg_0, c="C3", zorder=5, lw=1, label="g=0")
    axes[0].plot(test_angles[mask], hg_0[mask], c="C0", zorder=10)

    hg_0 = hg(np.deg2rad(test_angles), g=-0.2)

    axes[0].plot(test_angles, hg_0, c="C3", zorder=5, lw=1, ls="--", label="g=-0.2")
    axes[0].plot(test_angles[mask], hg_0[mask], c="C0", zorder=10)

    hg_0 = hg(np.deg2rad(test_angles), g=0.4)

    axes[0].plot(test_angles, hg_0, c="C3", zorder=5, lw=1, ls=":", label="g=0.4")
    axes[0].plot(test_angles[mask], hg_0[mask], c="C0", zorder=10)

    axes[0].fill_betweenx(
        axes[0].get_ylim(), min_angle, max_angle, c="C0", alpha=0.2, zorder=0
    )
    axes.format(xlabel="Scattering angle (°)", ylabel="Polarization fraction")

    fig.savefig(paths.figures / "test_spf_coverage.pdf", bbox_inches="tight")
