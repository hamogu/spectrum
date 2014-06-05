Documentation
=============

This package provides tools to manipulate astronomical spectra. 
It is based on the astropy project and will hopefully be affiliated with this
project in the future. Note that the astropy project already hosts a
a package called specutils. There are some differences in design compared
with specutils, most importantly the :class:`spectrum.Spectrum` in this 
package is based on :class:`astropy.table.Table`. This design decision:

- makes is easy to do things with the spectrum (slice, coadd, join, etc.)
- requires that the WCS is provided as a table column, not as a polynomial
  of some sort.

Therefore, we currently develop this package independently, but we aim to 
integrate them eventually.

.. toctree::
  :maxdepth: 2

  spectrum/index.rst
