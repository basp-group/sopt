
# SOPT: Sparse OPTimisation
------------------------------------------------------------------

## Description

SOPT is a C++ package to perform Sparse OPTimisation. It solves a variety of sparse regularisation
problems, including the SARA algorithm. Prototype Matlab implementations of various algorithms are
also included.

## Contributors

SOPT was initially created by Rafael Carrillo, Jason McEwen and Yves Wiaux but major contributions
have since been made by a number of others. The full list of contributors is as follows:

* [Rafael E. Carrillo](http://people.epfl.ch/rafael.carrill)
* [Jason D. McEwen](http://www.jasonmcewen.org)
* [Yves Wiaux](http://basp.eps.hw.ac.uk)
* [Vijay Kartik](https://people.epfl.ch/vijay.kartik)
* [Mayeul d'Avezac](https://github.com/mdavezac)
* [Luke Pratley](https://about.me/luke.pratley)
* [David Perez-Suarez](https://dpshelio.github.io)

## References
When referencing this code, please cite our related papers:

1. R. E. Carrillo, J. D. McEwen, D. Van De Ville, J.-P. Thiran, and Y. Wiaux.  "Sparsity averaging
   for compressive imaging", IEEE Signal Processing Letters, 20(6):591-594, 2013,
   [arXiv:1208.2330](http://arxiv.org/abs/arXiv:1208.2330)
1. A. Onose, R. E. Carrillo, A. Repetti, J. D. McEwen, J.-P. Thiran, J.-C. Pesquet, and Y. Wiaux.
   "Scalable splitting algorithms for big-data interferometric imaging in the SKA era". Mon. Not.
   Roy. Astron. Soc., 462(4):4314-4335, 2016,
   [arXiv:1601.04026](http://arxiv.org/abs/arXiv:1601.04026)

## Webpage

http://basp-group.github.io/sopt/

## Installation

### C++ pre-requisites and dependencies

- [CMake](http://www.cmake.org/): a free software that allows cross-platform compilation
- [tiff](http://www.libtiff.org/): Tag Image File Format library
- [OpenMP](http://openmp.org/wp/): Optional. Speeds up some of the operations.
- [UCL/GreatCMakeCookOff](https://github.com/UCL/GreatCMakeCookOff): Collection of cmake recipes.
  Downloaded automatically if absent.
- [Eigen 3](http://eigen.tuxfamily.org/index.php?title=Main_Page): Modern C++ linear algebra.
  Downloaded automatically if absent.
- [spdlog](https://github.com/gabime/spdlog): Optional. Logging library. Downloaded automatically if
  absent.
- [philsquared/Catch](https://github.com/philsquared/Catch): Optional - only for testing. A C++
  unit-testing framework. Downloaded automatically if absent.
- [google/benchmark](https://github.com/google/benchmar): Optional - only for benchmarks. A C++
  micro-benchmarking framework. Downloaded automatically if absent.

### Python pre-requisites and dependencies

- [numpy](http://www.numpy.org/): Fundamental package for scientific computing with Python
- [scipy](https://www.scipy.org/): User-friendly and efficient numerical routines such as routines
  for numerical integration and optimization
- [pandas](http://pandas.pydata.org/): library providing high-performance, easy-to-use data
  structures and data analysis tools
- [cython](http://cython.org/): Makes writing C extensions for Python as easy as Python itself.
  Downloaded automatically if absent.
- [pytest](http://doc.pytest.org/en/latest/): Optional - for testing only. Unit-testing framework
  for python. Downloaded automatically if absent and testing is not disabled.

### Installing Sopt

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

If you have any questions or comments, feel free to contact Rafael Carrillo or Jason McEwen, or add
an issue in the [issue tracker](https://github.com/basp-group/sopt/issues).

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
