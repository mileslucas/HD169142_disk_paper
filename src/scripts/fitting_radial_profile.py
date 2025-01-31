import numpy as np
import paths
import tqdm
import pandas as pd

# import pocoMC
import pocomc as pc
from scipy import stats
import matplotlib.pyplot as plt


# Set the random seed.
np.random.seed(169142)

dates = ("20230707", "20240729")
names = ["F610", "F670", "F720", "F760"]
dist = 114.5
pxscale = 5.9e-3
iwas = {"20230707": 105, "20240727": 59, "20240728": 59, "20240729": 59}


def double_powerlaw_ring(radius, r0, amp, alpha1, alpha2):
    dist1 = (radius / r0) ** (2 * alpha1)
    dist2 = (radius / r0) ** (2 * alpha2)
    return amp * np.sqrt(2) / np.sqrt(1 / dist1 + 1 / dist2)

def gaussian_ring(radius, r0, amp, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sq_mahab_dist = (radius - r0)**2 / sigma**2
    return amp * np.exp(-0.5 * sq_mahab_dist)


def model_two_double_powerlaw_rings(radius, params):
    r_inner = params[0]
    amp_inner = params[1]
    alpha1_inner = params[2]
    alpha2_inner = params[3]

    r_outer = params[4 + 0]
    amp_outer = params[4 + 1]
    alpha1_outer = params[4 + 2]
    alpha2_outer = params[4 + 3]

    inner_ring = double_powerlaw_ring(
        radius, r_inner, amp_inner, alpha1_inner, alpha2_inner
    )
    outer_ring = double_powerlaw_ring(
        radius, r_outer, amp_outer, alpha1_outer, alpha2_outer
    )
    return inner_ring + outer_ring


def model_gaussian_plus_double_powerlaw_rings(radius, params):
    r_inner = params[0]
    amp_inner = params[1]
    fwhm_inner = params[2]

    r_outer = params[3 + 0]
    amp_outer = params[3 + 1]
    alpha1_outer = params[3 + 2]
    alpha2_outer = params[3 + 3]

    inner_ring = gaussian_ring(
        radius, r_inner, amp_inner, fwhm_inner
    )
    outer_ring = double_powerlaw_ring(
        radius, r_outer, amp_outer, alpha1_outer, alpha2_outer
    )
    return inner_ring + outer_ring


def fit_model(radius, data, err):
    # Define the dimensionality of our problem.
    def llhood(X):
        model = model_two_double_powerlaw_rings(radius, X)
        return -0.5 * np.nansum((model - data) ** 2 / err**2)

    X0 = [21, data.max(), 5, -5, 60, data.max() * 0.25, 3, -1]
    print(f"{llhood(X0):=}")
    plot_model(radius, data, X0)
    prior = pc.Prior(
        [
            stats.halfnorm(X0[0], 5), # radius_inner
            stats.halfnorm(X0[1], 0.1), # amplitude_inner
            stats.uniform(1e-2, 10), # alpha_inner
            stats.uniform(-10, 10 - 1e-2), # beta_inner
            stats.halfnorm(X0[4], 20), # radius outer
            stats.halfnorm(X0[5], 0.1), # amplitude outer
            stats.uniform(1e-2, 20), # alpha_outer
            stats.uniform(-10, 10 - 1e-2), # beta_outer
        ]
    )

    # initialize sampler
    sampler = pc.Sampler(prior=prior, likelihood=llhood, random_state=169142)

    # start sampling
    sampler.run()

    # Get the results
    samples, logl, logp = sampler.posterior(resample=True)

    posterior = {"samples": samples, "logl": logl, "logp": logp}

    return posterior


def plot_model(radius, data, params):
    inner_model = double_powerlaw_ring(radius, *params[:4])
    outer_model = double_powerlaw_ring(radius, *params[4:])
    plt.scatter(radius, data, c="C0")
    plt.plot(radius, inner_model, c="C1")
    plt.plot(radius, outer_model, c="C1")
    plt.plot(radius, inner_model + outer_model, c="C2")
    plt.xlim(0, 150)
    # plt.yscale("log")
    plt.show(block=True)
    plt.clf()


if __name__ == "__main__":
    for i, date in enumerate(tqdm.tqdm(dates)):
        # load data
        table = pd.read_csv(
            paths.data / date / f"{date}_HD169142_vampires_radial_profiles.csv"
        )
        groups = table.groupby("filter")
        profiles = []
        errs = []
        for wl_idx, (filt_name, group) in enumerate(tqdm.tqdm(groups)):
            radius = group["radius(au)"].values
            profiles.append(group["Qphi"])
            errs.append(group["Qphi_err"])
        mean_profile = np.nanmean(profiles, axis=0)
        mean_err = np.sqrt(np.nanmean(np.power(errs, 2), axis=0))
        # mask out everything outside of IWA to OWA
        # mask out the satellite spots
        iwa_mask = radius > iwas[date] / 1e3 * dist
        owa_mask = radius <= 150
        satspot_mask = (radius > 30) & (radius < 47)

        mask = iwa_mask & owa_mask & ~satspot_mask
        # X0 = [19, 0.12, 3, -4, 67, 0.4 * 0.12, 3, -1]

        posterior_dict = fit_model(radius[mask], mean_profile[mask], mean_err[mask])
        np.savez(paths.data / date / f"{date}_HD169142_vampires_radial_profile_posteriors.npz", **posterior_dict)
        best_fit = np.median(posterior_dict["samples"], axis=0)
        plot_model(radius, mean_profile, best_fit)
