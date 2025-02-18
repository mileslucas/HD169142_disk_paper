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

def bootstrap_argmax_and_max(xs, signal, error, N=10000):
    # assume normally distributed errorss
    signal_samples = signal[None, :] + np.random.randn(N, len(signal)) * error[None, :]

    max_xs = []
    max_values = []
    for _signal in signal_samples:
        max_idx = np.nanargmax(_signal)
        max_xs.append(xs[max_idx])
        max_values.append(_signal[max_idx])

    max_x = np.mean(max_xs, axis=0)
    max_x_std = np.std(max_xs, axis=0)

    max_value = np.mean(max_values, axis=0)
    max_value_std = np.std(max_values, axis=0)


    return max_x, max_x_std, max_value, max_value_std


def bootstrap_argmin_and_min(xs, signal, error, N=10000):
    # assume normally distributed errorss
    signal_samples = signal[None, :] + np.random.randn(N, len(signal)) * error[None, :]

    min_xs = []
    min_values = []
    for _signal in signal_samples:
        min_idx = np.nanargmin(_signal)
        min_xs.append(xs[min_idx])
        min_values.append(_signal[min_idx])

    min_x = np.mean(min_xs, axis=0)
    min_x_std = np.std(min_xs, axis=0)

    min_value = np.mean(min_values, axis=0)
    min_value_std = np.std(min_values, axis=0)


    return min_x, min_x_std, min_value, min_value_std

