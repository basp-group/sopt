#ifndef SOPT_WAVELETS_WAVELETS_H
#define SOPT_WAVELETS_WAVELETS_H

#include <Eigen/Core>
#include "traits.h"

namespace sopt { namespace wavelets {

# define SOPT_WAVELET_MACRO(NAME, SIZE)          \
  /** Selects a specific wavelet type */         \
  struct NAME ## Tag {                           \
    /** Coefficients for the wavelet */          \
    const static Eigen::Matrix<t_real, SIZE, 1> coefficients; \
    const static Eigen::Matrix<t_real, SIZE, 1> high_pass;    \
    const static Eigen::Matrix<t_real, SIZE, 1> low_pass;     \
  };

  SOPT_WAVELET_MACRO(Dirac, 1);
  SOPT_WAVELET_MACRO(Daubechies1,   2);
  SOPT_WAVELET_MACRO(Daubechies2,   4);
  SOPT_WAVELET_MACRO(Daubechies3,   6);
  SOPT_WAVELET_MACRO(Daubechies4,   8);
  SOPT_WAVELET_MACRO(Daubechies5,  10);
  SOPT_WAVELET_MACRO(Daubechies6,  12);
  SOPT_WAVELET_MACRO(Daubechies7,  14);
  SOPT_WAVELET_MACRO(Daubechies8,  16);
  SOPT_WAVELET_MACRO(Daubechies9,  18);
  SOPT_WAVELET_MACRO(Daubechies10, 20);

# undef SOPT_WAVELET_MACRO

}}
#endif
