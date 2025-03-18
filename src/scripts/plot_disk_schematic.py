import paths
import proplot as pro
from matplotlib import patches
import numpy as np
from utils_plots import setup_rc

if __name__ == "__main__":
    setup_rc()

    width = 3.31314

    fig, ax = pro.subplots(width=f"{width}in")

    ## alma
    alma_patch_kwargs = dict(ec="0.2", fc="0.2", zorder=15)
    alma_text_kwargs = dict(c="k", fontsize=5, ha="center", bbox=dict(boxstyle="square", facecolor="w", edgecolor="k"), zorder=20)

    b0_width = 1.5 # au
    circ = patches.Ellipse((0, 0), width=b0_width * np.cos(np.deg2rad(35.2-12.45)), height=b0_width, angle=31.55-5.88, **alma_patch_kwargs)
    ax.add_patch(circ)
    ax.text(0, 5, "B0", va="bottom", **alma_text_kwargs)

    b1 = 32
    b1_width=b1 - 24
    ann = patches.Annulus((0, 0), r=b1, width=b1_width, **alma_patch_kwargs)
    ax.add_patch(ann)
    ax.text(26, 0, "B1", va="center", **alma_text_kwargs)


    b2_width=1.5
    b2 = 58 + b2_width/2
    ann = patches.Annulus((0, 0), r=b2, width=b2_width, **alma_patch_kwargs)
    ax.add_patch(ann)
    b2_rad = b2 - b2_width / 2
    _th = np.arctan2(5, b2_rad)
    ax.text(b2_rad * np.cos(_th), 5, "B2", va="center", **alma_text_kwargs)

    b3_width=1.8
    b3 = 65 + b3_width/2
    ann = patches.Annulus((0, 0), r=b3, width=b3_width, **alma_patch_kwargs)
    ax.add_patch(ann)
    b3_rad = b3 - b3_width / 2
    _th = np.arctan2(5, b3_rad)
    ax.text(b3_rad * np.cos(_th), -5, "B3", va="center", **alma_text_kwargs)

    b4_width=3.4
    b4 = 77 + b4_width/2
    ann = patches.Annulus((0, 0), r=b4, width=b4_width, **alma_patch_kwargs)
    ax.add_patch(ann)
    ax.text(77, 0, "B4", va="center", **alma_text_kwargs)


    ## gaps
    ang = -45
    r_d1 = 9
    arc_length = np.deg2rad(90) * r_d1
    wedge = np.rad2deg(arc_length / r_d1)
    ts = np.linspace(ang - wedge/2, ang + wedge/2, 100)
    ys_d1 = r_d1 * np.sin(np.deg2rad(ts))
    xs_d1 = r_d1 * np.cos(np.deg2rad(ts))
    ax.plot(xs_d1, ys_d1, ls=":", lw=1, c="0.2")
    ax.text(r_d1 * np.cos(np.deg2rad(ang)), r_d1 * np.sin(np.deg2rad(ang)), "D1", va="center", **alma_text_kwargs)

    r_d2 = 41
    arc_length = np.deg2rad(140) * r_d1
    wedge = np.rad2deg(arc_length / r_d2)
    ts = np.linspace(ang - wedge/2, ang + wedge/2, 100)
    ys_d2 = r_d2 * np.sin(np.deg2rad(ts))
    xs_d2 = r_d2 * np.cos(np.deg2rad(ts))
    ax.plot(xs_d2, ys_d2, ls=":", lw=1, c="0.2")
    ax.text(r_d2 * np.cos(np.deg2rad(ang)), r_d2 * np.sin(np.deg2rad(ang)), "D2", va="center", **alma_text_kwargs)

    r_d3 = (b2 + b3 - b3_width) / 2
    wedge = np.rad2deg(arc_length / r_d3)
    ts = np.linspace(ang - wedge/2, ang + wedge/2, 100)
    ys_d3 = r_d3 * np.sin(np.deg2rad(ts))
    xs_d3 = r_d3 * np.cos(np.deg2rad(ts))
    ax.plot(xs_d3, ys_d3, ls=":", lw=1, c="0.2")
    ax.text(r_d3 * np.cos(np.deg2rad(ang + 2)), r_d3 * np.sin(np.deg2rad(ang + 2)), "D3", va="center", **alma_text_kwargs)

    r_d4 = (b3 + b4 - b4_width) / 2
    wedge = np.rad2deg(arc_length / r_d4)
    ts = np.linspace(ang - wedge/2, ang + wedge/2, 100)
    ys_d4 = r_d4 * np.sin(np.deg2rad(ts))
    xs_d4 = r_d4 * np.cos(np.deg2rad(ts))
    ax.plot(xs_d4, ys_d4, ls=":", lw=1, c="0.2")
    ax.text(r_d4 * np.cos(np.deg2rad(ang - 2)), r_d4 * np.sin(np.deg2rad(ang - 2)), "D4", va="center", **alma_text_kwargs)

    ## scattered light
    text_kwargs = dict(c="C0", fontsize=6, ha="center", fontweight="bold", bbox=dict(boxstyle="square", facecolor="w", edgecolor="C0"), zorder=20)


    b1_width=8
    b1 = 21 + b1_width / 2
    ann = patches.Annulus((0, 0), r=b1, width=b1_width, ec="C0", fc="C0", alpha=0.9, zorder=10)
    ax.add_patch(ann)
    ax.text(0, 23, "Inner", va="center", **text_kwargs)


    b3 = 100
    b3_width=(b3 - (40.9 + 17.3/2))
    ann = patches.Annulus((0, 0), r=b3, width=b3_width, ec="C0", fc="C0", alpha=0.4, zorder=10)
    ax.add_patch(ann)    
    ax.text(66, 23, "Outer", va="center", **text_kwargs)


    # protoplanet b
    r = 36.4 # au
    theta0 = 34.9
    y = r * np.sin(np.deg2rad(theta0 + 90))
    x = r * np.cos(np.deg2rad(theta0 + 90))
    ax.scatter(x, y, marker="+", c="0.2", s=30, lw=0.5)
    ax.text(
        x + 3, y + 3, "b", fontsize=6, ha="center", va="center", c="0.2"
    )

    # scale bar
    bar_width_au = 20 # au
    bar_location = 60
    ax.plot([bar_location, bar_location + bar_width_au], [-95, -95], lw=2, c="k")
    ax.text(bar_location + bar_width_au/2, -95 + 3, f"{bar_width_au} au", ha="center", va="bottom")


    ax.format(xlim=(-110, 110), ylim=(-110, 110), grid=False)
    ax.axis("off")

    fig.savefig(paths.figures / "HD169142_schematic.pdf", bbox_inches="tight")
