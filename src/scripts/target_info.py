from dataclasses import dataclass
import numpy as np

# calculate the stellar mass aggregated from a couple sources
masses = [
    (2.11, 0.14), # B22
    (1.393, 0.2), # Y21
    (2.0, 0.13), # R22
    (1.65, 0.2), # B06
]

def mean_and_std(values):
    _vals = np.array([v[0] for v in values])
    _errs = np.array([v[1] for v in values])
    N = len(values)
    mean = np.mean(_vals)
    std = np.std(_vals) / np.sqrt(N)
    rms = np.sqrt(np.sum(_errs**2)) / N
    err = np.hypot(std, rms)
    return mean, err

_mass, _mass_err = mean_and_std(masses)
ages = [
    (6, 3, 6), # G07
    (8.98, 3.9, 11.02), # R22
    (12, 12 - 4, 12 + 8),
]

def uneven_mean_and_std(values):
    _vals = np.array([v[0] for v in values])
    _errs_low = np.array([v[1] for v in values])
    _errs_hi = np.array([v[2] for v in values])
    _errs_ave = np.sqrt(_errs_low * _errs_hi)
    N = len(values)
    mean = np.mean(_vals)
    std = np.std(_vals) / np.sqrt(N)
    rms = np.sqrt(np.sum(_errs_ave**2)) / N
    err = np.hypot(std, rms)
    return mean, err

_stellar_age, _stellar_age_err = uneven_mean_and_std(ages)

@dataclass(repr=True)
class TargetInfo:
    name = "HD169142"
    plx = 8.7053e-3  # " +- 0.0268e-3
    inclination = 12.45 # deg
    pos_angle = 5.88 # deg, location of far side minor axis
    stellar_mass =  _mass # Msun
    stellar_mass_err =  _mass_err # Msun
    stellar_age = _stellar_age # Myr
    stellar_age_err = _stellar_age_err # Myr

    @property
    def dist_pc(self):
        return 1 / self.plx

target_info = TargetInfo()

if __name__ == "__main__":
    print(target_info.stellar_mass, target_info.stellar_mass_err)
    print(target_info.stellar_age, target_info.stellar_age_err)