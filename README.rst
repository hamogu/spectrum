====================
The spectrum package
====================

If you run into any problems, don't hesitate to ask for help on the
astropy-dev mailing list!

This package provides a template for packages that are affiliated with the
project. This package design mirrors the layout of the main
`Astropy`_ repository, as well as reusing much of the helper code used to
organize `Astropy`_.  The instructions below describe how to take this
template and adjust it for your particular affiliated package.

This package provides tools to manipulate astronomical spectra. 
It is based on the `Astropy`_  project and will hopefully be affiliated with this
project in the future. Note that the `Astropy`_ project already hosts a
a package called specutils. There are some differences in design compared
with specutils, most importantly the  class ``Spectrum`` in this 
package is based on ``astropy.table.Table``. This design decision:

- makes is easy to do things with the spectrum (slice, coadd, join, etc.)
- requires that the WCS is provided as a table column, not as a polynomial
  of some sort.

Therefore, we currently develop this package independently, but we aim to 
integrate them eventually.


.. _Astropy: http://www.astropy.org/

