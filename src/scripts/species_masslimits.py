import matplotlib.pyplot as plt
import numpy as np
from species import SpeciesInit
from species.data.database import Database
from species.read.read_isochrone import ReadIsochrone
from species.plot.plot_spectrum import plot_spectrum
from species.plot.plot_color import plot_color_magnitude
import tqdm
import paths
import pandas as pd

def contrast_to_mag(contrast, stellar_mag=5.66):
    delta_mag = -2.5 * np.log10(contrast)
    return stellar_mag + delta_mag

def get_closest_mass(limit, mags, masses):
    _idx = np.where(mags <= limit)[0][0]
    return masses[_idx]

if __name__ == "__main__":
    SpeciesInit()

    database = Database()
    database.add_isochrones(model='atmo')
    read_iso = ReadIsochrone(tag='atmo-ceq')
    masses = np.linspace(1e-1, 5, 100)
    # Wallack+2023
    contrast_02 = 1.2e-3 # 0.2" 5sigma contrast
    contrast_04 = 1.41e-4 # 0.4" 5sigma contrast
    contrast_03 = np.sqrt(contrast_02 * contrast_04) # 0.3" 5sigma contrast from geometric mean

    contrast_limits = contrast_to_mag(np.array((contrast_02, contrast_03)))
    ages = [9-5, 9, 9+5]#, 1]
    rows = []
    for age in tqdm.tqdm(ages, desc="Testing different system ages"):
        iso_box = read_iso.get_isochrone(age=age, masses=masses, filter_mag='MKO_Lp')
        iso_box.open_box()
        for contrast_limit in contrast_limits:
            mass_lim = get_closest_mass(contrast_limit, iso_box.magnitude, iso_box.mass)
            rows.append({
                "age": age,
                "contrast(mag_Lp)": contrast_limit,
                "mass": mass_lim
            })

            # plt.plot(iso_box.mass, iso_box.magnitude, label=f'Age = {iso_box.age} Myr')

            # plt.axhline(contrast_limit, c="k", lw=1)
            # plt.axvline(mass_lim, c="k", lw=1)

            # plt.xlabel(r'Mass ($M_\mathrm{Lp}$)', fontsize=14)
            # plt.ylabel(iso_box.filter_mag, fontsize=14)
            # plt.gca().invert_yaxis()
            # plt.legend(fontsize=14)
            # plt.show(block=True)

    df = pd.DataFrame(rows)
    df.to_csv(paths.data / "HD169142_species_mass_limits.csv", index=False)
    
    print(df.groupby("contrast(mag_Lp)")["mass"].agg(["min", "max", "mean", "std"]))