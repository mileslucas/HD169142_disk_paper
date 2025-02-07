import paths
import proplot as pro
from matplotlib import patches
from utils_plots import setup_rc

if __name__ == "__main__":
    setup_rc()

    width = 3.31314

    fig, ax = pro.subplots(width=f"{width}in")

    ## alma
    alma_patch_kwargs = dict(ec="0.2", fc="0.2")
    alma_text_kwargs = dict(c="k", fontsize=6, ha="center", bbox=dict(boxstyle="square", facecolor="w", edgecolor="k"), zorder=20)

    b0_width = 1.5 # au
    circ = patches.Circle((0, 0), radius=b0_width, **alma_patch_kwargs)
    ax.add_patch(circ)
    ax.text(0, 5, "B0", va="bottom", **alma_text_kwargs)

    b1_width=6
    b1 = 26 + b1_width/2
    ann = patches.Annulus((0, 0), r=b1, width=b1_width, **alma_patch_kwargs)
    ax.add_patch(ann)
    ax.text(26, 0, "B1", va="center", **alma_text_kwargs)


    b2_width=1.5
    b2 = 57 + b2_width/2
    ann = patches.Annulus((0, 0), r=b2, width=b2_width, **alma_patch_kwargs)
    ax.add_patch(ann)
    ax.text(55, 5, "B2", va="center", **alma_text_kwargs)

    b3_width=1.8
    b3 = 64 + b3_width/2
    ann = patches.Annulus((0, 0), r=b3, width=b3_width, **alma_patch_kwargs)
    ax.add_patch(ann)
    ax.text(64, -5, "B3", va="center", **alma_text_kwargs)

    b4_width=3.4
    b4 = 77 + b4_width/2
    ann = patches.Annulus((0, 0), r=b4, width=b4_width, **alma_patch_kwargs)
    ax.add_patch(ann)
    ax.text(77, 0, "B4", va="center", **alma_text_kwargs)


    ## scattered light
    text_kwargs = dict(c="C0", fontsize=6, ha="center", fontweight="bold", bbox=dict(boxstyle="square", facecolor="w", edgecolor="C0"), zorder=20)


    b1_width=6
    b1 = 20 + b1_width / 2
    ann = patches.Annulus((0, 0), r=b1, width=b1_width, ec="C0", fc="C0", alpha=0.9, zorder=10)
    ax.add_patch(ann)
    ax.text(0, -23, "Inner", va="center", **text_kwargs)


    b3 = 100
    b3_width=45
    ann = patches.Annulus((0, 0), r=b3, width=b3_width, ec="C0", fc="C0", alpha=0.4, zorder=10)
    ax.add_patch(ann)    
    ax.text(0, -77.5, "Outer", va="center", **text_kwargs)


    # scale bar
    bar_width_au = 20 # au
    bar_location = 60
    ax.plot([bar_location, bar_location + bar_width_au], [-95, -95], lw=2, c="k")
    ax.text(bar_location + bar_width_au/2, -95 + 3, f"{bar_width_au} au", ha="center", va="bottom")


    ax.format(xlim=(-110, 110), ylim=(-110, 100), grid=False)
    ax.axis("off")

    fig.savefig(paths.figures / "HD169142_schematic.pdf", bbox_inches="tight")
