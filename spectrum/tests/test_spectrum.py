from copy import deepcopy

import numpy as np

import astropy.units as u

from ..spectrum import coadd_simple, COSspectrum


class SetupData(object):
    @property
    def a(self):
        if not hasattr(self, '_a'):
            wave = np.arange(5000, 5500) * u.Angstrom
            flux = (1. + np.random.random(len(wave))) * u.erg/u.Angstrom/u.second
            spec = COSspectrum({'WAVE':wave, 'FLUX': flux}, dispersion='WAVE')
            spec.meta['ORIGIN'] = 'Example'
            self._a = spec
        return self._a


    @property
    def b(self):
        if not hasattr(self, '_b'):
            wave = np.arange(5000, 5500) * u.Angstrom
            flux = (1. + np.random.random(len(wave))) * u.erg/u.Angstrom/u.second
            error = 0.1 * np.ones_like(flux)
            spec = COSspectrum({'WAVE':wave, 'FLUX': flux, 'ERROR': error}, 
                               dispersion='WAVE', uncertainty='ERROR')
            spec.meta['ORIGIN'] = 'Example'
            self._b = spec
        return self._b


    @property
    def c(self):
        if not hasattr(self, '_c'):
            wave = np.arange(5000, 5010) * u.Angstrom
            flux = (np.arange(10)) * u.erg/u.Angstrom/u.second
            error = 0.1 * flux + 0.1 * u.erg/u.Angstrom/u.second
            spec = COSspectrum({'WAVE':wave, 'FLUX': flux, 'ERROR': error}, 
                               dispersion='WAVE', uncertainty='ERROR')
            spec.meta['ORIGIN'] = 'Example'
            self._c = spec
        return self._c


    @property
    def d(self):
        if not hasattr(self, '_d'):
            wave = np.arange(5020, 5030) * u.Angstrom
            flux = (np.arange(10)) * u.erg/u.Angstrom/u.second
            error = 0.1 * flux + 0.1 * u.erg/u.Angstrom/u.second
            spec = COSspectrum({'WAVE':wave, 'FLUX': flux, 'ERROR': error}, 
                               dispersion='WAVE', uncertainty='ERROR')
            spec.meta['ORIGIN'] = 'Example'
            self._d = spec
        return self._d


class TestSpectrum(SetupData):
    def test_slicedisp(self):
        out = self.b.slice_disp([5050*u.Angstrom, 5100*u.Angstrom])
        assert out.disp[0] == 5050 * u.Angstrom
        assert out.disp[-1] == 5100 * u.Angstrom
        assert out.meta['ORIGIN'] == 'Example'
        assert np.all(out.flux == self.b.flux[50:101])
        assert np.all(out.error == self.b.error[50:101])


    def test_slicerv(self):
        out = self.a.slice_rv([-300*u.km/u.s, +3e7*u.cm/u.s], 520.*u.nm)
        assert out.disp[0] == 5195 * u.Angstrom
        assert out.disp[-1] == 5205 * u.Angstrom
        assert out.meta['ORIGIN'] == 'Example'
        
    
    def test_binup(self):
        out = self.b.bin_up(2)
        assert np.all(out.disp == np.arange(5000.5,5500, 2) * u.AA)
        assert np.all(np.abs(out.error- 0.1/np.sqrt(2)* u.erg/u.Angstrom/u.second) < 1e-6* u.erg/u.Angstrom/u.second)
        assert np.abs(out.flux[0] / ((self.b.flux[0]+self.b.flux[1])/2.) -1.) < 1e-6
        assert out.meta['ORIGIN'] == 'Example'

    def test_bin_up_noexact(self):
        out = self.b[:11]
        assert len(out) == 11
        assert len(out.bin_up(2)) == 5
        assert len(out.bin_up(3)) == 3
        assert len(out.bin_up(11)) == 1


    def test_interpol(self):
        wave = np.arange(500.05, 500.9, 0.1) * u.nm
        out = self.c.interpol(wave)
        assert np.all(np.abs(out.disp/(np.arange(5000.5, 5009)*u.AA) - 1) < 1e-6)
        assert np.all(np.abs(out.flux/(np.arange(0.5, 9)* u.erg/u.Angstrom/u.second)-1)<1e-6)
        assert out.meta['ORIGIN'] == 'Example'


    def test_interpol_nearest(self):
        wave = np.arange(500.01, 501, 0.1) * u.nm
        out = self.c.interpol(wave, kind='nearest', bounds_error=False)
        assert np.all(out.disp == wave)
        assert np.all(out.flux[:-1] == self.c.flux[:-1])
        assert np.isnan(out.flux[-1])  # out of bound interpolation
        assert np.all(out.error[:-1] == self.c.error[:-1])


    def test_shiftrv(self):
        wave = np.array([100, 300]) * u.nm
        flux = np.array([1,1]) * u.W / u.second
        spec = COSspectrum({'WAVE':wave, 'FLUX': flux}, dispersion='WAVE')
        spec.shift_rv(300*u.km/u.s)
        assert np.abs(spec.disp[0]/(100.1*u.nm)-1) < 1e-6
        assert np.abs(spec.disp[1]/(300.3*u.nm)-1) < 1e-6

class TestCoadd(SetupData):
    def test_coadd_one(self):
        out = coadd_simple([self.c])
        assert np.all(self.c.flux == out.flux)
        assert np.all(self.c.error == out.error)


    def test_coadd_simple(self):
        out = coadd_simple([self.a, self.a, self.a])
        assert np.all(np.abs(out.flux/self.a.flux-1) < 1e-6)
        assert np.all(out.disp == self.a.disp)


    def test_coadd_simple_error(self):
        out = coadd_simple([self.b, self.b, self.b])
        assert np.all(np.abs(out.flux/self.b.flux-1) < 1e-6)
        assert np.all(np.abs(out.error/(0.1/np.sqrt(3)*u.erg/u.AA/u.second)-1) < 1e-6)
        assert np.all(out.disp == self.b.disp)

    
    def test_coadd_disp(self):
        disp = np.arange(5050, 5100) * u.Angstrom
        out = coadd_simple([self.a, self.a, self.a], dispersion=disp)
        assert np.all(out.disp == disp)
        assert np.all(np.abs(out.flux/self.a.flux[50:100]-1) < 1e-6)


    def test_coadd_nonoverlapping(self):
        disp = np.arange(5000,5030) * u.Angstrom
        out = coadd_simple([self.c, self.d], dispersion=disp, bounds_error=False)
        assert np.all(self.c.flux == out.flux[:len(self.c)])
        assert np.all(self.c.error == out.error[:len(self.c)])
        assert np.all(self.d.flux == out.flux[-len(self.d):])
        assert np.all(self.d.error == out.error[-len(self.d):])
