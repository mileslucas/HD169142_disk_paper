import paths
import diskmap
from astropy.io import fits
import numpy as np
from astropy.nddata import Cutout2D
import tqdm
from target_info import target_info
from pathlib import Path
from instrument_info import zimpol_info, irdis_info, gpi_info


def crop(data, window):
    cy, cx = np.array(data.shape[-2:]) / 2 - 0.5
    cutout = Cutout2D(data, (cx, cy), window)
    return cutout.data


def get_diskmap_outputs(data, name, pxscale, rmax=700):
    mapping = diskmap.DiskMap(
        data,
        pixscale=pxscale,
        distance=target_info.dist_pc,
        image_type="polarized",
        inclination=target_info.inclination,
        pos_angle=target_info.pos_angle,
    )

    name.parent.mkdir(parents=True, exist_ok=True)
    mapping.map_disk(power_law=(0, 0, 0), radius=(1, rmax, rmax))
    mapping.r2_scaling(r_max=rmax)
    # mapping.phase_function(radius=(15, 45), n_phase=36, g=0)
    mapping.deproject_disk()
    mapping.total_intensity(g=0)
    mapping.write_output(filename=str(name))
    # degrees east of north, in disk coordinates
    az_deg = np.mod(-90 - np.rad2deg(mapping.azimuth) + target_info.pos_angle, 360)
    fits.writeto(f"{name}_azimuth.fits", az_deg, overwrite=True)


def process_vampires_data(folder, filename):
    vampires_stokes_cube, vampires_hdr = fits.getdata(
        filename,
        header=True,
    )
    crop_size = 400

    vampires_Qphi = crop(vampires_stokes_cube[4], crop_size)
    get_diskmap_outputs(
        vampires_Qphi, name=paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Qphi", pxscale=vampires_hdr["PXSCALE"] / 1e3, rmax=200
    )
    vampires_Uphi = crop(vampires_stokes_cube[5], crop_size)
    get_diskmap_outputs(
        vampires_Uphi, name=paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Uphi", pxscale=vampires_hdr["PXSCALE"] / 1e3, rmax=200
    )

def process_zimpol_data(folder: str):
    zpl_Qphi, _ = fits.getdata(
        filename = paths.data / folder / "Qphi.fits",
        header=True,
    )
    tmp_name = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Qphi"
    get_diskmap_outputs(zpl_Qphi, name=tmp_name, pxscale=zimpol_info.pxscale, rmax=300)

    zpl_Uphi, _ = fits.getdata(
        filename = paths.data / folder / "Uphi.fits",
        header=True,
    )
    tmp_name = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Uphi"
    get_diskmap_outputs(zpl_Uphi, name=tmp_name, pxscale=zimpol_info.pxscale, rmax=300)

def process_irdis_data(folder: str, filename: Path):
    cube, _ = fits.getdata(
        filename,
        header=True,
    )
    irdis_Qphi = crop(cube[1], 500)
    tmp_name = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Qphi"
    get_diskmap_outputs(irdis_Qphi, name=tmp_name, pxscale=irdis_info.pxscale, rmax=700)
    irdis_Uphi = crop(cube[2], 500)
    tmp_name = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Uphi"
    get_diskmap_outputs(irdis_Uphi, name=tmp_name, pxscale=irdis_info.pxscale, rmax=700)


def process_naco_data(folder: str):
    naco_Qphi, naco_Qphi_hdr = fits.getdata(
        filename = paths.data / folder / "coadded" / "Q_phi.fits", header=True, ext=("Q_PHI_CTC_IPS", 1)
    )
    naco_Qphi = crop(naco_Qphi, 120)

    tmp_name = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Qphi"
    get_diskmap_outputs(
        naco_Qphi, name=tmp_name, pxscale=naco_Qphi_hdr["ESO INS PIXSCALE"], rmax=300
    )

    naco_Uphi, naco_Uphi_hdr = fits.getdata(
        filename = paths.data / folder / "coadded" / "U_phi.fits", header=True, ext=("U_PHI_CTC_IPS", 1)
    )
    naco_Uphi = crop(naco_Uphi, 120)

    tmp_name = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Uphi"
    get_diskmap_outputs(
        naco_Uphi, name=tmp_name, pxscale=naco_Uphi_hdr["ESO INS PIXSCALE"], rmax=300
    )


def process_gpi_data(folder: str, filename: Path):
    cube, hdr = fits.getdata(
        filename,
        header=True,
    )
    gpi_Qphi = cube[1]
    tmp_name = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Qphi"
    get_diskmap_outputs(gpi_Qphi, name=tmp_name, pxscale=gpi_info.pxscale, rmax=400)
    gpi_Uphi = cube[2]
    tmp_name = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Uphi"
    get_diskmap_outputs(gpi_Uphi, name=tmp_name, pxscale=gpi_info.pxscale, rmax=400)

def process_alma_data(folder: str, filename: Path):
    alma_data, alma_hdr = fits.getdata(filename, header=True)
    alma_pxscale = np.abs(alma_hdr["CDELT1"]) * 3.6e3 # arcsec / px
    
    tmp_name = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_I"
    get_diskmap_outputs(alma_data, name=tmp_name, pxscale=alma_pxscale, rmax=350)


def process_charis_data(folder: str):
    cube, hdr = fits.getdata(
        filename=paths.data / folder / f"{folder}_HD169142_stokes_collapsed.fits",
        header=True,
    )
    charis_Qphi = crop(cube[4], 140)
    tmp_name = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Qphi"
    get_diskmap_outputs(charis_Qphi, name=tmp_name, pxscale=15.16e-3, rmax=300)

    charis_Uphi = crop(cube[5], 140)
    tmp_name = paths.data / folder / "diskmap" / f"{folder}_HD169142_diskmap_Uphi"
    get_diskmap_outputs(charis_Uphi, name=tmp_name, pxscale=15.16e-3, rmax=300)

if __name__ == "__main__":
    folders = [
        "20120726_NACO_H",
        "20140425_GPI_J",
        "20150503_IRDIS_J",
        "20150710_ZIMPOL_VBB",
        "20170918_ALMA_1.3mm",
        "20180715_ZIMPOL_VBB",
        "20210906_IRDIS_Ks",
        "20230604_CHARIS_JHK",
        "20230707_VAMPIRES_MBI",
        "20240729_VAMPIRES_MBI",
    ]
    for folder in tqdm.tqdm(folders):
        if "VAMPIRES" in folder:
            date = folder.split("_")[0]
            filename = (
                paths.data / folder / "optimized" / f"{date}_HD169142_vampires_stokes_cube_optimized.fits"
            )
            process_vampires_data(folder, filename)
        elif "NACO" in folder:
            process_naco_data(folder)
        elif "ZIMPOL" in folder:
            process_zimpol_data(folder)
        elif "IRDIS" in folder:
            filename = paths.data / folder / f"{folder}_HD169142_stokes_cube.fits"
            process_irdis_data(folder, filename)
        elif "GPI" in folder:
            filename = paths.data / folder / f"{folder}_HD169142_stokes_cube.fits"
            process_gpi_data(folder, filename)
        elif "ALMA" in folder:
            filename = paths.data / folder / "HD169142.selfcal.concat.GPU-UVMEM.centered_mJyBeam.fits"
            process_alma_data(folder, filename)
        elif "CHARIS" in folder:
            process_charis_data(folder)
        else:
            print(f"Unrecognized folder: {folder=}")

    
    