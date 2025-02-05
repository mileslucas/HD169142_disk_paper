import proplot as pro
import paths
import pandas as pd
from astropy.time import Time
from scipy import signal
import numpy as np
from utils_crosscorr import phase_correlogram


if __name__ == "__main__":
    pro.rc["font.size"] = 8
    pro.rc["label.size"] = 8
    pro.rc["title.size"] = 9
    pro.rc["figure.dpi"] = 300
    pro.rc["cycle"] = "ggplot"


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



    axes[0].plot(x, A, label="A", c="C0")
    axes[0].plot(x, B, label="B", c="C3")

    lags, xcorr = phase_correlogram(B, A)
    norm_xcorr = xcorr / np.nanmax(xcorr)
    norm_lags = lags / 5 * spacing

    axes[1].plot(norm_lags, norm_xcorr, c="C0")
    max_corr_ind = np.nanargmax(norm_xcorr)
    axes[1].axvline(lag_known, c="C0", lw=1, alpha=0.6)
    axes[1].axvline(norm_lags[max_corr_ind], c="C0", lw=1, ls="--", alpha=0.6)
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
        ax.axvline(0, c="0.3", lw=1, zorder=0)

    # axes[-1].legend(ncols=1, fontsize=8, order="F")
    axes[0].legend()
    axes[0].format(
        xlabel="x (rad)",
        ylabel="y",
        title="Data"
    )
    axes[1].format(
        xlabel="Offset (rad)",
        # xlocator=1,
        title="Phase cross-correlation"
    )

    fig.savefig(
        paths.figures / "test_profiles_crosscorr.pdf",
        bbox_inches="tight",
        dpi=300,
    )


    ## 2
    
