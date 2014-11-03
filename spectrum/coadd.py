import numpy as np
from .spectrum import Spectrum


def wave_little_interpol(wavelist):
    '''Make a wavelengths array for merging echelle orders with little interpolation.

    In echelle spectra we often have the situation that neighboring orders overlap
    a little in wavelength space::

        aaaaaaaaaaaa
                 bbbbbbbbbbbbb
                          ccccccccccccc

    When merging those spectra, we want to keep the original wavelength grid where possible.
    This way, we only need to interpolate on a new wavelength grid where different orders
    overlap (here ``ab`` or ``bc``) and can avoid the dangers of flux interpolation in
    those wavelength region where only one order contributes.

    This algorithm has limitations, some are fundamental, some are just due to the 
    implementation and may be removed in future versions:

    - The resulting grid is **not** equally spaced, but the step size should not vary too much.
    - The wavelength arrays need to be sorted in increasing order.
    - There has to be overlap between every order and every order has to have some overlap
      free region in the middle.

    Parameters
    ----------
    wavelist : list of 1-dim ndarrays
        input list of wavelength

    Returns
    -------
    waveout : ndarray
        wavelength array that can be used to co-adding all echelle orders.
    '''
    mins = np.array([min(w) for w in wavelist])
    maxs = np.array([max(w) for w in wavelist])
    
    if np.any(np.argsort(mins) != np.arange(len(wavelist))):
        raise ValueError('List of wavelengths must be sorted in increasing order.')
    if np.any(np.argsort(mins) != np.arange(len(wavelist))):
        raise ValueError('List of wavelengths must be sorted in increasing order.')
    if not np.all(maxs[:-1] > mins[1:]):
        raise ValueError('Not all orders overlap.')
    if np.any(mins[2:] < maxs[:-2]):
        raise ValueError('No order can be completely overlapped.')

    waveout = [wavelist[0][wavelist[0]< mins[1]]]
    for i in range(len(wavelist)-1):
        #### overlap region ####
        # No assumptions on how bin edges of different orders match up
        # overlap start and stop are the last and first "clean" points.
        overlap_start = np.max(waveout[-1])
        overlap_end = np.min(wavelist[i+1][wavelist[i+1] > maxs[i]])
        # In overlap region patch in a linear scale with slightly different step.
        dw = overlap_end - overlap_start
        step = 0.5*(np.mean(np.diff(wavelist[i])) + np.mean(np.diff(wavelist[i+1])))
        n_steps = np.int(dw / step + 0.5)

        wave_overlap = np.linspace(overlap_start + step,  overlap_end - step, n_steps - 1)
        waveout.append(wave_overlap)

        #### next region without overlap ####
        if i < (len(wavelist) -2):  # normal case
            waveout.append(wavelist[i+1][(wavelist[i+1] > maxs[i]) & (wavelist[i+1]< mins[i+2])])
        else:                       # last array - no more overlap behind that
            waveout.append(wavelist[i+1][(wavelist[i+1] > maxs[i])])

    return np.hstack(waveout)



def coadd_simple(spectra, dispersion=None, **kwargs):
    '''A simple way to coadd several spectra

    All spectra are interpolated to ``dispersion`` and for each wavelengths bin the
    mean over all n spectra is calculated. The reported uncertainty is just the mean
    of all uncertainties scaled by 1/sqrt(n).
    Nan values are ignored in the calculation of the mean. Thus, this method can be used
    if not all spectra have the same wavelength range. Supply the keyword
    ``bounds_error=False``, so that the interpolation returns ``nan`` outside the range
    covered by the spectrum.
        

    Parameters
    ----------
    spectra : list of :class:`~spectrum.Spectrum` instances
        spectra to be averaged
    dispersion : :class:`~astropy.quantity.Quantity`
        dispersion axis of the new spectrum. If ``None`` the dispersion axis of the 
        first spectrum in ``spectra`` is used.
    
    All other parameters are passed to :meth:`~spectrum.Spectrum.interpol`.

    Returns
    -------
    spec : :class:`~spectrum.Spectrum`

    See also
    --------
    coadd_errorweighted
    '''
    if dispersion is None:
        dispersion = spectra[0].disp
    fluxes = np.zeros((len(spectra), len(dispersion)))
    # since numpy operation will destroy the flux unit, need to convert here by hand 
    # until that is fixed (fluxes can be NDdata once that works with quantities)
    fluxunit = spectra[0].flux.unit
    errors = np.zeros_like(fluxes)
    for i, s in enumerate(spectra):
        s_new = s.interpol(dispersion, **kwargs)
        fluxes[i,:] = s_new.flux.to(fluxunit)
        if (errors is None) or s.uncertainty is None:
            errors = None
        else:
            errors[i,:] = s_new.error.to(fluxunit)

    # This can be simplified considerably as soon as masked quantities exist.
    fluxes = np.ma.fix_invalid(fluxes)
    fluxmask = np.ma.getmaskarray(fluxes)
    fluxes = fluxes.mean(axis=0).filled(fill_value=np.nan) * fluxunit
    
    if errors is None:
        return Spectrum(data=[dispersion, fluxes],
                           names=[spectra[0].dispersion, 'FLUX'],
                           dispersion=spectra[0].dispersion,
                           )
    else:
        errors =  np.ma.fix_invalid(errors)
        if fluxmask.sum() > 0:
            errors[fluxmask] = np.ma.masked  # In case flux has more entries masked than error
        errors = (errors.mean(axis=0)/np.sqrt((~np.ma.getmaskarray(errors)).sum(axis=0))).filled(np.nan)*fluxunit
        return Spectrum(data=[dispersion, fluxes, errors],
                           names=[spectra[0].dispersion, 'FLUX', spectra[0].uncertainty], 
                       dispersion=spectra[0].dispersion, uncertainty=spectra[0].uncertainty,
                           )



def coadd_errorweighted(spectra, dispersion=None, **kwargs):
    '''A simple way to coadd several spectra

    All spectra are interpolated to ``dispersion`` and for each wavelengths bin the
    mean over all n spectra is calculated. The reported uncertainty is just the mean
    of all uncertainties scaled by 1/sqrt(n).
    Nan values are ignored in the calculation of the mean. Thus, this method can be used
    if not all spectra have the same wavelength range. Supply the keyword
    ``bounds_error=False``, so that the interpolation returns ``nan`` outside the range
    covered by the spectrum.
        

    Parameters
    ----------
    spectra : list of :class:`~spectrum.Spectrum` instances
        spectra to be averaged
    dispersion : :class:`~astropy.quantity.Quantity`
        dispersion axis of the new spectrum. If ``None`` the dispersion axis of the 
        first spectrum in ``spectra`` is used.
    
    All other parameters are passed to :meth:`~spectrum.Spectrum.interpol`.

    Returns
    -------
    spec : :class:`~spectrum.Spectrum`

    See also
    --------
    coadd_simple
    '''
    if dispersion is None:
        dispersion = spectra[0].disp
    fluxes = np.ma.zeros((len(spectra), len(dispersion)))
    # since numpy operation will destroy the flux unit, need to convert here by hand 
    # until that is fixed (fluxes can be NDdata once that works with quantities)
    fluxunit = spectra[0].flux.unit
    errors = np.zeros_like(fluxes)
    for i, s in enumerate(spectra):
        s_new = s.interpol(dispersion, **kwargs)
        fluxes[i,:] = s_new.flux.to(fluxunit)
        if s.uncertainty is None:
            raise ValueError('s.uncertainty needs to be set for every spectrum')
        else:
            errors[i,:] = s_new.error.to(fluxunit)

    # First, make sure there is no flux defined if there is no error.
    errors = np.ma.fix_invalid(errors)
    if np.ma.is_masked(errors):
        fluxes[errors.mask] = np.ma.masked
    # This can be simplified considerably as soon as masked quantities exist.
    fluxes = np.ma.fix_invalid(fluxes)
    # There are no masked quantities yet, so make sure they are filled here.
    fluxes = np.ma.average(fluxes, axis=0, weights = 1./errors**2.).filled(np.nan) * fluxunit
    errors = np.sqrt(1. / np.ma.sum(1./errors**2., axis=0).filled(np.nan)) * fluxunit
    # check explicitly here every time, because that makes hard to find bugs and 
    # astropy quantity behaviour might change.
    assert not np.ma.isMaskedArray(fluxes)
    assert not np.ma.isMaskedArray(errors) 

    return Spectrum(data=[dispersion, fluxes, errors],
                       names=[spectra[0].dispersion, 'FLUX', spectra[0].uncertainty], 
                   dispersion=spectra[0].dispersion, uncertainty=spectra[0].uncertainty,
                       )

