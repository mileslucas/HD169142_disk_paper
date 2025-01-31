import paths
import corner
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 6

dates = ("20230707", "20240729")

param_names = [r"$r_{inner}$", r"$A_{inner}$", r"$\alpha_{inner}$", r"$\beta_{inner}$"]
param_names += [r"$r_{outer}$", r"$A_{outer}$", r"$\alpha_{outer}$", r"$\beta_{outer}$"]

if __name__ == "__main__":
    for i, date in enumerate(dates):
        traces = np.load(
            paths.data
            / date
            / f"{date}_HD169142_vampires_radial_profile_posteriors.npz"
        )
        fig = plt.figure(figsize=(7, 7), dpi=100)
        corner.corner(
            traces["samples"],
            labels=param_names,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            fig=fig
        )
        fig.savefig(
            paths.figures / f"{date}_HD169142_vampires_corner.pdf",
            bbox_inches="tight"
        )
        plt.close(fig)
