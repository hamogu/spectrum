import copy
import warnings

import numpy as np
from scipy import interpolate
# could implement fallback to np.interp if scipy is not available

from astropy import table
import astropy.units as u
import astropy.constants as const
from astropy.extern import six
from astropy.modeling import models, fitting

# These are functions in order to emphasize that they do stuff to spectra,
# it's not a property of the spectrum.
# Or put everything for single spectrum in spectrum class?
# bin_up is not for flux spectrum.
# should use np.sum() for count number spectrum -> Inheritance diagram
# instead of slice disp, should I just support that in __getitem__ (if input is a 
# wavelength) or is that too magical?
# currently use doppler_optical - make different options (e.g. variable)
# add way to change units, e.g. wave in km/s, flux at different scale
# Allow None for slice to go to end of array? Or just tell people to use np/inf with proper units?
# change COS spectrum init, so that I can initialize a spectrum form a spectrum (need to copy disp etc)
# investigate why deepcopy(sepctrum) does not copy dispersion and how to fix that
# implement devision to return "flux ratio object" (which could just be spectrum with a different flux name)?
# make slice_XXX accept list of  ranges, e.g. left and right of line.


def xcorr(speclist, waverange):
    '''cross-correlate spectral segments in a certain range

    More stuff to be considered:
   
    find / report some diagnostic for the case that the correlation does not work well
    make ``shift`` array an input, make step size input
    find way to make faster
    rename speclist to spectra for consistency with coadd
    unit tests


    Parameters
    ----------
    speclist : list
        List of :class:`~spectrum.spectrum.Spectrum` with 'WAVELENGTH' and 'FLUX' columns
    waverange : [float, float]
        lower and upper end of wavelength range
  
    Returns
    -------
    res : array of len(speclist)
        shift relative to first :class:`~spectrum.spectrum.Spectrum` in speclist
       
    '''
    specbase = speclist[0].slice_disp(waverange)
    res = np.zeros(len(speclist))
    shift = np.arange(-20.,+20) * u.km/u.s # see below - careful when changing
    cor = np.zeros(shift.shape)
    for j in range(len(speclist)):
        testspec = copy.deepcopy(speclist[j])
        testspec.dispersion = speclist[j].dispersion 
        testspec.shift_rv(-21*u.km/u.s)
        for i in range(len(cor)):
            testspec.shift_rv(1*u.km/u.s) # see above
            cor[i] = (specbase.flux * testspec.interpol(specbase.disp).flux).sum().value
        g = models.Gaussian1D(amplitude=np.max(cor), mean=0., stddev=0.5)
        pfit = fitting.LevMarLSQFitter()
        new_model = pfit(g, shift, cor)
        res[j] = new_model.mean.value
    return res * u.km/u.s


class Spectrum(table.Table):
    '''
    To make more general:
    make "dispersion" or something similar and allow to set that as an alias for
    what I currently call wavelength
    '''
    fluxname = 'FLUX'  # default to None, here, make setter in init for general
    dispersion = None  # set to default 'WAVELENGTH' for specific COSspectrum
    uncertainty = None  # can be removed when uncertainties are part of NDData

    def __init__(self, *args, **kwargs):
        if 'dispersion' in kwargs:
            self.dispersion = kwargs.pop('dispersion')
        if 'uncertainty' in kwargs:
            self.uncertainty = kwargs.pop('uncertainty')

        super(Spectrum, self).__init__(*args, **kwargs)

    def _copy_property_names(self, spec):
        spec.fluxname = copy.copy(self.fluxname)
        spec.dispersion = copy.copy(self.dispersion)
        spec.uncertainty = copy.copy(self.uncertainty)

    def _new_from_slice(self, slice_):
        spec = super(Spectrum, self)._new_from_slice(slice_)
        self._copy_property_names(spec)
        return spec

    def __getitem__(self, item):
        spec = super(Spectrum, self).__getitem__(item)
        if isinstance(item, (tuple, list)) and all(isinstance(x, six.string_types)
                                                     for x in item):
            self._copy_property_names(spec)
        return spec


    @classmethod
    def read(cls, filename):
        '''specific to COS - can be made into a Reader'''
        data = []
        
        tab = table.Table.read(filename)
        for c in tab.columns:
            if len(tab[c].shape) == 2:
                data.append(table.Column(data=tab[c].data.flatten(),
                                         unit=tab[c].unit,
                                         name=c))
        flattab = cls(data, meta=tab.meta, dispersion='WAVELENGTH')
        # COS is not an echelle spectrograph, so there never is any overlap
        flattab.sort('WAVELENGTH')
        return flattab

    @property
    def flux(self):
        if self.fluxname is None:
            raise ValueError('Need to set spectrum.fluxname="NAME".')
        return self[self.fluxname].data * self[self.fluxname].unit


    @property
    def disp(self):
        # Currently table columns are not quantities.
        # see issue astropy/#2486
        if self.dispersion is None:
            raise ValueError('Need to set spectrum.dispersion="NAME".')
        return self[self.dispersion].data * self[self.dispersion].unit

    @property
    def error(self):
        # Can be removed when NDData really supports uncertainties
        if self.uncertainty is None:
            raise ValueError('Need to set spectrum.uncertainty="NAME".')
        return self[self.uncertainty].data * self[self.uncertainty].unit


    def _slice_disp(self, bounds, equivalencies = []):
        b0 = bounds[0].to(self.disp.unit, equivalencies=equivalencies)
        b1 = bounds[1].to(self.disp.unit, equivalencies=equivalencies)
        ind = (self.disp >= b0) & (self.disp <= b1)
        return self[ind]


    def slice_disp(self, bounds):
        '''Return a portion of the spectrum between given dispersion values

        Parameters
        ----------
        bounds : list of two quantities
            [lower bound, upper bound] in dispersion of spectrally equivalent unit
        
        Returns
        -------
        spec : :class:`~spectrum.Spectrum`
            spectrum that is limited to the range from bound[0] to bound[1]

        See also
        --------
        slice_rv
        '''
        equil = u.spectral()
        return self._slice_disp(bounds, equivalencies=equil)


    def slice_rv(self, bounds, rest):
        '''Return a portion of the spectrum between given radial velocity values

        Parameters
        ----------
        bounds : list of two quantities
            [lower bound, upper bound] in radial velocity
        rest : :class:`~astropy.quantity.Quantity`
            Rest wavelength/frequency of spectral feature
        
        Returns
        -------
        spec : :class:`~spectrum.Spectrum`
            spectrum that is limited to the range from bound[0] to bound[1]

        See also
        --------
        slice_disp
        '''
        equil = u.spectral()
        equil.extend(u.doppler_optical(rest))
        return self._slice_disp(bounds, equivalencies=equil)


    def normalize(self, model, **kwargs):
        #fit model and normalize flux and error
        # Makes sense here or have user interact with flux directly?
        raise NotImplementedError


    def shift_rv(self, rv):
        '''Shift spectrum by rv
        
        Parameters
        ----------
        rv : :class:`~astropy.quantity.Quantity`
            radial velocity (positive value will red-shift the spectrum, negative
            value will blue-shift the spectrum)
        '''
        self[self.dispersion] = (self.disp.to(u.m, equivalencies=u.spectral()) * (
                1.*u.dimensionless_unscaled+rv/const.c)).to(
                self.disp.unit, equivalencies=u.spectral()).value


    # overload add, substract, divide to interpol automatically?
        
    def bin_up(self, factor, other_cols={}):
        '''bin up an array by factor ``factor``
   
        If the number of elements in spectrum is not n * factor with n=1,2,3,...
        the remaining bins at the end are discarded.
        By itself, this function knows how to deal with dispersion, flux and uncertainty.
        The parameter `other_cols` can be used to specify how to bin up other columns in 
        the spectrum. Columns with no rule for binning up are discarded.
   
        Parameters
        ----------
        x : array
            data to be binned
        factor : integer
            binning factor
        other_cols : dictionary
            The keys in this dictionary are the names of columns in the spectrum, its values
            are functions that define how to bin up other columns (e.g. for a column that holds
            a mask or data quality flag it does not make sense to calculate the mean).
            The function is called on an array of shape [N, factor] and needs to return 
            a [N] array. For example `other_cols={'quality': lambda x: np.max(x, axis=1)}`
            would assign the new bin the maximum of all quality values in the original bins
            that contribute to the new bin.

        Returns
        -------
        spec : :class:`~spectrum.Spectrum`
            A new spectrum object.
        '''
        n = len(self) // factor
        # makes a copy incl. all metadata and column formats
        # Which cols to keep?
        keepcols = set([x for x in [self.dispersion, self.fluxname, self.uncertainty] if x is not None])
        keepcols = keepcols.union(set(other_cols.keys()))
        spec = self[list(keepcols)][:n*factor:factor]
        spec[self.dispersion] = (self.disp[:n*factor].reshape((n, factor))).mean(axis=1)
        if self.uncertainty is None:
            spec[self.fluxname] = (self.flux[:n*factor].reshape((n, factor))).mean(axis=1)
        else:
            f, e = np.ma.average(self.flux[:n*factor].reshape((n, factor)), weights=1./(self.error[:n*factor].reshape((n, factor)))**2., axis=1, returned=True)
            spec[self.fluxname] = f
            spec[self.uncertainty] = (1./e)**0.5
        for col in other_cols:
            spec[col] = other_cols[col](self[col][:n*factor].reshape((n, factor)))
        return spec


    def interpol(self, new_dispersion, **kwargs):
        '''Interpolate a spectrum onto a new dispersion axis.

        Parameters
        ----------
        new_dispersion : :class:`~astropy.quantity.Quantity`
           The new dispersion axis.

        All other keywords are passed directly to scipy.interpolate.interp1d.

        Returns
        -------
        spec : :class:`~spectrum.Spectrum`
            A new spectrum.
        '''
        new_disp = new_dispersion.to(self.disp.unit, equivalencies=u.spectral())

        f_flux = interpolate.interp1d(self.disp, self.flux, **kwargs)
        newflux = f_flux(new_disp)

        names = [self.dispersion, self.fluxname]
        vals = [new_disp, newflux]

        if self.uncertainty is not None:
            warnings.warn('The uncertainty column is interpolated.' + 
                         'Bins are no longer independent and might require scaling.' +
                         'It is up to the user the decide if the uncertainties are still meaningful.')
            names.append(self.uncertainty)
            f_uncert = interpolate.interp1d(self.disp, self.error, **kwargs)
            vals.append(f_uncert(new_disp))

        # TBD Add other columns that should be interpolated to names, vals here
        # Add switch or keyword to select them

        newcols = []
        for name, val in zip(names, vals):
            col = self[name]
            newcols.append(col.__class__(data=val, name=name, 
                                         description=col.description, unit=col.unit,
                                         format=col.format, meta=col.meta))
        return self.__class__(newcols, meta=self.meta, dispersion=self.dispersion, 
                              uncertainty=self.uncertainty)
        
    def crosscorrelate(self, dispersion, flux):
        '''or as a module level function?
        Do full thing here with steps, return best fit etc? or only calculate one spesific
        step?
        '''
        raise NotImplementedError
