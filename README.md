
# SOPT: Sparse OPTimisation
------------------------------------------------------------------

## Description

SOPT is a C++ package to perform Sparse OPTimisation. It solves a variety of sparse regularisation
problems, including the SARA algorithm. Prototype Matlab implementations of various algorithms are
also included.

## Creators

* [Rafael E. Carrillo](http://people.epfl.ch/rafael.carrillo)
* [Jason D. McEwen](http://www.jasonmcewen.org)
* [Yves Wiaux](http://basp.eps.hw.ac.uk)

## Contributors

* Mayeul d'Avezac
* Vijay Kartik

## References
When referencing this code, please cite our related paper:

    [1] R. E. Carrillo, J. D. McEwen, D. Van De Ville, J.-P. Thiran,
    and Y. Wiaux.  "Sparsity averaging for compressive imaging", IEEE
    Signal Processing Letters, Vol. 20, No. 6, pp. 591-594, 2013
    (arXiv:1208.2330).

## Webpage

http://basp-group.github.io/sopt/

## Installation

The build system uses [CMake](https://cmake.org/).

The C++ library also requires [Eigen 3](http://eigen.tuxfamily.org/index.php?title=Main_Page),
[spdlog](https://github.com/gabime/spdlog), and [tiff](http://www.remotesensing.org/libtiff/)
(optionally). The first two will be downloaded automatically if the build system cannot find them.

Additionally, the Python bindings require [numpy](http://www.numpy.org/),
[scipy](https://www.scipy.org/), [pandas](http://pandas.pydata.org/), [cython](http://cython.org/)
(during build only), and [pytest](http://doc.pytest.org/en/latest/) (for testing only). The last
two will be downloaded automatically if not found. However, the user will have to take care of
installing the first three (via pip, or conda, for instance).

Once the dependencies are present, the program can be built with:

```
cd /path/to/code
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

To test everything went all right:

```
cd /path/to/code/build
ctest .
```

To install in directory `/X`, with libraries going to `X/lib`, python modules to
`X/lib/pythonA.B/site-packages/sopt`, etc, do:

```
cd /path/to/code/build
cmake -DCMAKE_INSTALL_PREFIX=/X ..
make install
```


## Support

If you have any questions or comments, feel free to contact Rafael Carrillo or Jason McEwen, or to
add an issue in the [issue tracker](https://github.com/astro-informatics/sopt/issues).

## Notes

The code is given for educational purpose. For the matlab version of the code see the folder matlab.

## License

    SOPT: Sparse OPTimisation package
    Copyright (C) 2013 Rafael Carrillo, Jason McEwen, Yves Wiaux
    
    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version.
    
    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details (LICENSE.txt).
    
    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301, USA.
