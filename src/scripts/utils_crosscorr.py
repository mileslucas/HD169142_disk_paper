import numpy as np
from target_info import target_info

_DEG_PER_PX = 5

def phase_correlogram(signal, ref_signal):

    im_fft = np.fft.fft(signal)
    ref_fft = np.fft.fft(ref_signal)

    cross_power_spectrum = im_fft * ref_fft.conj()
    cross_correlation = np.fft.ifft(cross_power_spectrum)

    correlogram = np.real(np.fft.fftshift(cross_correlation))

    lags_inds = np.arange(-len(signal) // 2, len(ref_signal) // 2)
    lags_degs = lags_inds * _DEG_PER_PX# / oversample
    return lags_degs, correlogram



if __name__ == "__main__":
    from astropy.io import fits
    import paths

    folders = [
        "20120726_NACO",
        "20140425_GPI",
        "20150503_IRDIS",
        "20150710_ZIMPOL",
        "20180715_ZIMPOL",
        "20210906_IRDIS",
        "20230707_VAMPIRES",
        "20240729_VAMPIRES",
    ]

    pxscales = {
        "20120726_NACO": 27e-3,
        "20140425_GPI": 14.14e-3,
        "20150503_IRDIS": 12.25e-3,
        "20150710_ZIMPOL": 3.6e-3,
        "20180715_ZIMPOL": 3.6e-3,
        "20230707_VAMPIRES": 5.9e-3,
        "20210906_IRDIS": 12.25e-3,
        "20240729_VAMPIRES": 5.9e-3,
    }

    data = fits.getdata(paths.data / "20140425_GPI" / "20140425_GPI_HD169142_Qphi_polar.fits")
    data2 = fits.getdata(paths.data / "20210906_IRDIS" / "20210906_IRDIS_HD169142_Qphi_polar.fits")


    _rs = np.linspace(15, 35, 50)
    _thetas = np.arange(0, 360, 5)
    common_thetas, common_rs = np.meshgrid(_thetas, _rs)
    
    im1 = reinterp(data, common_rs, common_thetas, pxscales["20140425_GPI"])
    im2 = reinterp(data2, common_rs, common_thetas, pxscales["20210906_IRDIS"])
    
    lags, corr = phase_correlogram(im2, im1)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1)

    kwargs= {"origin": "lower", "cmap": "magma"}
    # axes[0].imshow(im1, **kwargs)
    # axes[1].imshow(im2, **kwargs)
    print(lags.shape)
    print(corr.shape)
    axes.plot(lags, np.max(corr, axis=0))#, extent=(lags.min(), lags.max(), 0, corr.shape[0]), **kwargs)

    plt.show(block=True)

