import numpy as np

np.random.seed(169142)
def relative_deviation(signal, error):
    # step 1 calculate mean and standard error
    mean = np.mean(signal)
    std = np.std(signal)
    rmserr = np.sqrt(np.sum(np.power(error, 2)) / len(error)**2)
    stderr = std / np.sqrt(len(signal))
    mean_err = np.hypot(stderr, rmserr)

    D = signal / mean - 1
    D_err = np.hypot(error / mean, signal / mean**2 * mean_err)
    return D, D_err

def bootstrap_peak(xs, signal, error, N=10000):
    # assume normally distributed errorss
    signal_samples = signal[None, :] + np.random.randn(N, len(signal)) * error[None, :]

    peak_xs = []
    for _signal in signal_samples:
        peak_idx = np.nanargmax(_signal)

        peak_x = xs[peak_idx]
        peak_xs.append(peak_x)

    peak_x = np.mean(peak_xs, axis=0)
    peak_x_std = np.std(peak_xs, axis=0)


    return peak_x, peak_x_std
