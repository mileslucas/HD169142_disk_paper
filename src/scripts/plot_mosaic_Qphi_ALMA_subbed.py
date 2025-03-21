import paths
from astropy.io import fits
import numpy as np
import proplot as pro
from astropy.visualization import simple_norm
from utils_organization import label_from_folder, folders, pxscales
from utils_plots import setup_rc
from matplotlib import patches
from target_info import target_info
from utils_indexing import frame_radii
from astropy.stats import biweight_location
from skimage import filters
from vampires_dpp.image_processing import derotate_frame


def inner_ring_mask(frame, radii):
    rin_au = 15
    rout_au = 35
    rad_mask = (radii >= rin_au) & (radii <= rout_au)
    return np.where(rad_mask, frame, np.nan)

def create_radial_profile_image(frame, radii):
    output = np.zeros_like(frame)
    radii_ints = np.round(radii).astype(int)
    bins = np.arange(radii_ints.min(), radii_ints.max() + 1)
    count_map = {}
    for bin in bins:
        mask = (radii_ints == bin) & np.isfinite(frame)
        data = frame[mask]
        mean = np.nanmean(data)
        count_map[bin] = mean
    for ridx in range(output.shape[0]):
        for cidx in range(output.shape[1]):
            radius = radii[ridx, cidx]
            bin = np.round(radius).astype(int)
            value = count_map[bin]
            output[ridx, cidx] = value

    return output

def create_faux_adi_image(frame, N=100, pa_rot=60):
    pas = np.linspace(0, pa_rot, N)
    output = np.empty((N, *frame.shape), dtype=frame.dtype)
    for idx in range(N):
        output[idx] = derotate_frame(frame, -pas[idx])

    med_frame = np.nanmedian(output, axis=0, overwrite_input=True)
    return med_frame


if __name__ == "__main__":
    setup_rc()
    pro.rc["axes.facecolor"] = "w"
    pro.rc["axes.grid"] = False

    alma_data, alma_hdr = fits.getdata(paths.data / "20170918_ALMA_1.3mm" / "HD169142.selfcal.concat.GPU-UVMEM.centered_mJyBeam.fits", header=True)
    alma_pxscale = np.abs(alma_hdr["CDELT1"]) * 3.6e3 # arcsec / px
    alma_side_length = alma_data.shape[-1] * alma_pxscale / 2
    alma_ext = (alma_side_length, -alma_side_length, -alma_side_length, alma_side_length)
    alma_ys = np.linspace(alma_ext[0], alma_ext[1], alma_data.shape[0])
    alma_xs = np.linspace(alma_ext[2], alma_ext[3], alma_data.shape[1])

    fig, axes = pro.subplots(ncols=4, nrows=2, width="7in", hspace=1.75, wspace=0.5, spanx=False)


    for i, folder in enumerate(folders):
        Qphi_image_subbed = fits.getdata(paths.data / folder / f"{folder}_HD169142_Qphi_cADI_sim_r2_scaled.fits")
        radius_path = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Qphi_radius.fits"
        radius_map_au = fits.getdata(radius_path)

        Qphi_image_subbed = filters.gaussian(Qphi_image_subbed, 1)
        Qphi_image_masked = inner_ring_mask(Qphi_image_subbed, radius_map_au)

        side_length = Qphi_image_subbed.shape[-1] * pxscales[folder] / 2
        ext = (side_length, -side_length, -side_length, side_length)

        vmax = np.nanmax(Qphi_image_masked)
        norm = pro.DivergingNorm(vmin=-vmax, vmax=vmax)
        axes[i].imshow(Qphi_image_subbed, extent=ext, cmap="div", norm=norm, vmin=norm.vmin, vmax=norm.vmax)
        labels = label_from_folder(folder).split()
        axes[i].text(
            0.03, 1.01, labels[0],
            transform="axes",
            c="0.1",
            fontweight="bold",
            ha="left",
            va="bottom"
        )
        axes[i].text(
            0.99, 1.01, " ".join(labels[1:]),
            transform="axes",
            c="0.1",
            fontweight="bold",
            ha="right",
            va="bottom"
        )

        # patch = patches.Circle((0, 0), radius=17 / target_info.dist_pc, ec="w", fill=False, lw=1)
        # axes[i].add_patch(patch)

        # star position
        axes[i].scatter(0, 0, marker="+", lw=1, markersize=50, c="0.1")

        # PSF
        


    axes.format(
        xlim=(0.9, -0.9),
        ylim=(-0.9, 0.9),
        # xlocator=[0.6, 0.3, 0, -0.3, -0.6],
        # ylocator=[-0.6, -0.3, 0, 0.3, 0.6],
        # xlabel=r'$\Delta$RA (")',
        # ylabel=r'$\Delta$DEC (")',
        xlocator="none",
        ylocator="none"
    )

    # axes[1].format(yspineloc="none")

    fig.savefig(
        paths.figures / "HD169142_Qphi_mosaic_outer_subbed.pdf",
        bbox_inches="tight", dpi=300
    )
    levels = np.geomspace(0.05, np.nanmax(alma_data), 5)
    for ax in axes:
        ax.contour(alma_xs, alma_ys, alma_data, origin="lower", colors="0.1", alpha=0.5, levels=levels, lw=0.5)

    fig.savefig(
        paths.figures / "HD169142_Qphi_ALMA_mosaic_outer_subbed.pdf",
        bbox_inches="tight", dpi=300
    )