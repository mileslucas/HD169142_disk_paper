import numpy as np

rng = np.random.default_rng(169142)

_DEG_PER_PX = 5


def phase_correlogram(signal, ref_signal, degs_per_px=5):
    im_fft = np.fft.rfft(signal, norm="ortho")
    ref_fft = np.fft.rfft(ref_signal, norm="ortho")

    cross_power_spectrum = im_fft * ref_fft.conj()
    cross_correlation = np.fft.irfft(cross_power_spectrum, norm="ortho")
    # eps = np.finfo(cross_correlation.dtype).eps
    # cross_correlation /= np.maximum(np.abs(cross_correlation), 100 * eps)

    correlogram = np.real(np.fft.ifftshift(cross_correlation))
    # Compute the frequency bins
    lags_samples = np.arange(len(correlogram)) - len(correlogram) / 2
    lags_degs = lags_samples * degs_per_px

    return lags_degs, correlogram


# def phase_correlogram(signal, ref_signal, degs_per_px=5):
#     corr = np.correlate(signal, ref_signal, "full")

#     # Compute the frequency bins
#     lags_samples = np.arange(len(corr)) - len(corr)/2
#     lags_degs = lags_samples * degs_per_px
#     return lags_degs, corr


def bootstrap_phase_correlogram(signal, signal_err, ref_signal, ref_signal_err, N=10000, **kwargs):
    # assume normally distributed errorss
    signal_samples = rng.normal(loc=signal, scale=np.abs(signal_err), size=(N, *signal.shape))
    ref_signal_samples = rng.normal(
        loc=ref_signal, scale=np.abs(ref_signal_err), size=(N, *ref_signal.shape)
    )

    lags = []
    correlograms = []
    for _signal, _ref in zip(signal_samples, ref_signal_samples, strict=True):
        lags_degs, corr = phase_correlogram(_signal, _ref, **kwargs)
        lags.append(lags_degs)
        correlograms.append(corr)

    lags_mean = np.mean(lags, axis=0)
    correlogram_mean = np.mean(correlograms, axis=0)
    correlograms_std = np.std(correlograms, axis=0)
    return lags_mean, correlogram_mean, correlograms_std
