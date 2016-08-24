
                     SOPT: Sparse OPTimisation
  ----------------------------------------------------------------

DESCRIPTION
  SOPT is a C package to perform Sparse OPTimisation.  It solves a
  variety of sparse regularisation problems, including the SARA
  algorithm.  Prototype Matlab implementations of various algorithms
  are also included.  

CREATORS
  Rafael E. Carrillo (http://people.epfl.ch/rafael.carrillo)
  Jason D. McEwen (http://www.jasonmcewen.org)
  Yves Wiaux (http://basp.eps.hw.ac.uk)

CONTRIBUTORS
  Mayeul d'Avezac
  Vijay Kartik

EXPERIMENTS
  To run an in-painting reconstruction example, run the
  script sopt_demo1 in the bin folder.

  To run a Fourier sampling reconstruction example, run the
  script sopt_demo2 in the bin folder.

REFERENCES
  When referencing this code, please cite our related paper:
    [1] R. E. Carrillo, J. D. McEwen, D. Van De Ville, J.-P. Thiran,
    and Y. Wiaux.  "Sparsity averaging for compressive imaging", IEEE
    Signal Processing Letters, Vol. 20, No. 6, pp. 591-594, 2013
    (arXiv:1208.2330).

DOCUMENTATION
   See doc/html/index.html

WEBPAGE
   http://basp-group.github.io/sopt/

INSTALLATION
  SOPT requires the FFTW toolbox (www.fftw.org) and the TIFF toolbox
  (www.remotesensing.org/libtiff/) and can be built using cmake, for
  example, as follows:
  > mkdir build; cd build
  > cmake .. -DCMAKE_PREFIX_PATH='<fftw path>;<tiff path>'
  > make
  > make test

SUPPORT
  If you have any questions or comments, feel free to contact Rafael
  Carrillo or Jason McEwen.

NOTES
  The code is given for educational purpose. For the matlab version of
  the code see the folder matlab.

LICENSE
  SOPT: Sparse OPTimisation package
  Copyright (C) 2013 Rafael Carrillo, Jason McEwen, 
  Yves Wiaux

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
