import proplot as pro
import paths
import pandas as pd
from astropy.time import Time
from scipy import signal
import numpy as np


def time_from_folder(foldername: str):
    date_raw = foldername.split("_")[0]
    ymd = {
        "year": int(date_raw[:4]),
        "month": int(date_raw[4:6]),
        "day": int(date_raw[6:])
    }
    return Time(ymd, format="ymdhms")

# def cross_correlate(curve1, curve2, pxscale):
#     c1_pad
#     cross_corr = np.correlate(curve1, curve2, mode="full")
#     # Normalize correlation values
#     cross_corr /= (np.std(curve1) * np.std(curve2) * len(curve1))


#     # # cross_corr /= cross_corr.max()
#     lags_inds = signal.correlation_lags(len(curve1), len(curve2))
#     # # bin_width in azimuthal profile is 5 deg
#     lags_deg_per_yr = lags_inds * pxscale
#     # lags_deg_per_yr = np.arange(-len(curve1)//2, len(curve2)//2) * -pxscale
#     return lags_deg_per_yr, cross_corr

def cross_correlate(curve1, curve2, pxscale):
    fft1 = np.fft.fft(curve1)
    fft2 = np.fft.fft(curve2)


    R = fft1 * fft2.conj()
    r = np.fft.ifft(R)

    lags = np.arange(-len(curve1)//2, len(curve2)//2) * pxscale
    return lags, np.fft.fftshift(np.real(r))


if __name__ == "__main__":
    pro.rc["font.size"] = 9


    ## Plot and save
    fig, axes = pro.subplots(
        nrows=1, ncols=2, sharex=False, sharey=False
    )

    x = np.linspace(0, 2*np.pi, 1000)

    spacing = x[1] - x[0]

    A = np.sin(x)  # Base signal

    lag_known = 2  # Known shift in indices
    print(f"Lag: {lag_known}")

    B = np.sin(x - lag_known) + 0.1 * np.random.randn(len(x))  # Shifted + noise



    axes[0].plot(x, A, label="A")
    axes[0].plot(x, B, label="B")

    lags, xcorr = cross_correlate(B, A, spacing)
    norm_xcorr = xcorr / np.nanmax(xcorr)
    norm_lags = lags# * spacing - np.pi

    axes[1].plot(norm_lags, norm_xcorr)
    max_corr_ind = np.nanargmax(norm_xcorr)
    axes[1].axvline(lag_known, c="C0", lw=1, alpha=0.6)
    axes[1].axvline(norm_lags[max_corr_ind], c="C0", lw=1, ls="--", alpha=0.6)
    axes[1].axvline(0, c="0.3", lw=1, alpha=0.6, zorder=0)
    print(norm_lags[max_corr_ind])



    # axes[1, 0].text(
    #     0.03,
    #     0.95,
    #     "Inner",
    #     c="0.3",
    #     fontsize=9,
    #     transform="axes",
    #     ha="left",
    #     va="top",
    # )
    # axes[0, 0].text(
    #     0.03,
    #     0.95,
    #     "Outer",
    #     c="0.3",
    #     fontsize=9,
    #     transform="axes",
    #     ha="left",
    #     va="top",
    # )

    for ax in axes:
        ax.axhline(0, c="0.3", lw=1, zorder=0)

    # axes[-1].legend(ncols=1, fontsize=8, order="F")
    axes[0].legend()
    axes[0].format(
        xlabel="x (rad)",
        ylabel="y",
        title="Data"
    )
    axes[1].format(
        xlabel="Motion (rad)",
        # xlocator=1,
        title="Cross-correlation"
    )

    fig.savefig(
        paths.figures / "test_profiles_crosscorr.pdf",
        bbox_inches="tight",
        dpi=300,
    )


    ## 2
    
