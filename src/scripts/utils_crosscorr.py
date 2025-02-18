import numpy as np
from target_info import target_info

np.random.seed(169142)

_DEG_PER_PX = 5

  
def phase_correlogram(signal, ref_signal, degs_per_px=5):
    im_fft = np.fft.rfft(signal, norm="ortho")
    ref_fft = np.fft.rfft(ref_signal, norm="ortho")

    cross_power_spectrum = im_fft * ref_fft.conj()
    cross_correlation = np.fft.irfft(cross_power_spectrum, norm="ortho")

    correlogram = np.real(np.fft.fftshift(cross_correlation))
    # Compute the frequency bins
    lags_samples = np.arange(len(correlogram)) - len(correlogram)/2
    lags_degs = lags_samples * degs_per_px
    
    return lags_degs, correlogram


def bootstrap_phase_correlogram(signal, signal_err, ref_signal, ref_signal_err, N=10000, **kwargs):
    # assume normally distributed errorss
    signal_samples = signal[None, :] + np.random.randn(N, len(signal)) * signal_err[None, :]
    ref_signal_samples = ref_signal[None, :] + np.random.randn(N, len(signal)) * ref_signal_err[None, :]

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
