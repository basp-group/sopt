
                     SOPT: Sparse OPTimisation
                SARA algorithm for sparsity averaging
  ----------------------------------------------------------------

DESCRIPTION
  This is a Matlab implementation of the SARA algorithm presented in:
    [1] R. E. Carrillo, J. D. McEwen, D. Van De Ville, J.-P. Thiran,
    and Y. Wiaux.  "Sparsity averaging for compressive imaging", IEEE
    Signal Processing Letters, in press, 2013 (arXiv:1208.2330).

AUTHORS
  R. E. Carillo (http://people.epfl.ch/rafael.carrillo)
  G. Puy (http://people.epfl.ch/gilles.puy)
  J. D. McEwen (http://www.jasonmcewen.org)
  Y. Wiaux (http://people.epfl.ch/yves.wiaux)

EXPERIMENTS
  To test a reconstruction with spread spectrum measurements, run the
  script Experiment1.m. If you want to run the same experiment with
  the cameraman test image just change the variable image name to
  'cameraman_256.tiff'.

  To test a reconstruction with random Gaussian measurements, run the
  script Experiment2.m.

  To run the MRI example in the paper, run the script Experiment3.m.

REFERENCES
  When referencing this code, please cite our related paper:
    [1] R. E. Carrillo, J. D. McEwen, D. Van De Ville, J.-P. Thiran,
    and Y. Wiaux.  "Sparsity averaging for compressive imaging", IEEE
    Signal Processing Letters, in press, 2013 (arXiv:1208.2330).

DOCUMENTATION
   See doc/index.html

INSTALLATION 
  To run the experiments the CurveLab toolbox
  (www.curvelet.org) must be installed.  Otherwise the SOPT Matlab
  code should run as is, without any additional installation.

DOWNLOAD
  https://github.com/basp-group/sopt

SUPPORT
  If you have any questions or comments, feel free to contact Rafael
  Carrillo at: rafael {DOT} carrillo {AT} epfl {DOT} ch.

NOTES
  The code is not optimized and is given for educational purpose.
  This is a first Matlab implementation of the code. A more general
  and optimised C code is under development.

LICENSE
  SOPT: Sparse OPTimisation package
  Copyright (C) 2013 Rafael Carrillo, Gilles Puy, Jason McEwen, 
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
