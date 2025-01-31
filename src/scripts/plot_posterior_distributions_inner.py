import paths
import numpy as np
import arviz
import proplot as pro

dates = ("20230707", "20240729")
names = ["F610", "F670", "F720", "F760"]

param_names = ["r_inner", "A_inner", "g1_inner", "g2_inner"]
param_names += ["r_outer", "A_outer", "g1_outer", "g2_outer"]

fig, axes = pro.subplots(
    nrows=4, ncols=2, width="3.5in", refheight="1in", spany=False, wspace=0, hspace=0.5
)

cycle = pro.Colormap("fire")(np.linspace(0.4, 0.9, 4))

wavelengths = [612, 670, 719, 760]
span = wavelengths[-1] - wavelengths[0]
mid_wave = 0.5 * (wavelengths[0] + wavelengths[-1])
xmin = mid_wave - span / 2 * 1.3
xmax = mid_wave + span / 2 * 1.3

if __name__ == "__main__":
    for date_idx, date in enumerate(dates):
        for wl_idx, filt_name in enumerate(names):
            data = np.load(
                paths.data
                / date
                / f"{date}_HD169142_vampires_{filt_name}_radial_profile_posteriors.npz"
            )
            meds = np.median(data["samples"], axis=0)
            for param_idx in range(4):
                lo_67, hi_67 = arviz.hdi(data["samples"][:, param_idx], hdi_prob=0.67)
                lo_95, hi_95 = arviz.hdi(data["samples"][:, param_idx], hdi_prob=0.95)

                x_val = wavelengths[wl_idx]
                axes[param_idx, date_idx].scatter(
                    [x_val], [meds[param_idx]], ec=cycle[wl_idx], ms=10, fc="w", zorder=999
                )
                axes[param_idx, date_idx].plot(
                    [x_val, x_val], [lo_67, hi_67], lw=2, c=cycle[wl_idx]
                )
                axes[param_idx, date_idx].plot(
                    [x_val, x_val], [lo_95, hi_95], lw=1, c=cycle[wl_idx]
                )

    axes[0, 0].format(ylabel="radius (au)")
    axes[1, 0].format(ylabel="amplitude (Jy)")
    axes[2, 0].format(ylabel=r"$\gamma_{in}$")
    axes[3, 0].format(ylabel=r"$\gamma_{out}$")
    axes.format(
        xlim=(xmin, xmax),
        xlabel="Wavelength (nm)",
        toplabels=("2023/07/07", "2024/07/29"),
    )
    # axes[:-1, :].format(xspineloc="none")
    # axes[:, 1].format(yspineloc="none")
    fig.savefig(
        paths.figures / "HD169142_vampires_inner_ring_posteriors.pdf",
        bbox_inches="tight",
        dpi=300,
    )
