import proplot as pro
import paths
import pandas as pd
from astropy.time import Time
from scipy import signal
import numpy as np
from utils_crosscorr import phase_correlogram, bootstrap_phase_correlogram
from utils_errorprop import bootstrap_peak
from utils_plots import setup_rc

if __name__ == "__main__":
    setup_rc()

    ## Plot and save
    width = 3.31314
    fig, axes = pro.subplots(
        nrows=1, ncols=2, width=f"{width}in", wspace=0.75, share=False
    )

    x = np.linspace(0, 2*np.pi, 1000)

    spacing = x[1] - x[0]

    A = np.sin(x)  # Base signal
    A_err = 0.05 * np.random.randn(len(x))
    A += A_err # +noise
    
    lag_known = 2  # Known shift in indices
    print(f"Lag: {lag_known}")
    B = np.sin(x - lag_known)
    B_err = 0.05 * np.random.randn(len(x))
    B += B_err



    axes[0].plot(x, A, label="A", c="C0", lw=1)
    axes[0].plot(x, B, label="B", c="C3", lw=1)

    # lags, xcorr = phase_correlogram(B, A)
    lags, xcorr, xcorr_err = bootstrap_phase_correlogram(B, B_err, A, A_err)
    norm_val = np.nanmax(xcorr)
    norm_lags = lags / 5 * spacing

    axes[1].plot(norm_lags, xcorr, shadedata=xcorr_err, c="C0")
    peaklag, peaklagerr = bootstrap_peak(norm_lags, xcorr, xcorr_err)
    axes[1].axvline(lag_known, c="C0", lw=1, alpha=0.6)
    axes[1].axvline(peaklag, c="C0", lw=1, ls="--", alpha=0.6)
    print(f"{peaklag} +- {peaklagerr}")


    for ax in axes:
        ax.axhline(0, c="0.3", lw=1, zorder=0)
        ax.axvline(0, c="0.3", lw=1, zorder=0)

    axes[0].legend(ncols=1)
    axes[0].format(
        xlabel="x (rad)",
        ylabel="y",
        title="Data"
    )
    axes[1].format(
        xlabel="Offset (rad)",
        yformatter="none",
        # xlocator=1,
        title="Phase cross-correlation"
    )

    fig.savefig(
        paths.figures / "test_profiles_crosscorr.pdf",
        bbox_inches="tight",
        dpi=300,
    )


    ## 2
    
