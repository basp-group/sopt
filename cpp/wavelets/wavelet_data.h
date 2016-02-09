#ifndef SOPT_WAVELET_WAVELET_DATA_H
#define SOPT_WAVELET_WAVELET_DATA_H

#include "sopt/config.h"
#include "sopt/types.h"
#include "innards.impl.h"

namespace sopt {
namespace wavelets {

//! Holds wavelets coefficients
struct WaveletData {
  //! Type of the underlying scalar
  typedef t_real t_scalar;
  //! Type of the underlying vector
  typedef Array<t_real> t_vector;
  //! Wavelet coefficient per-se
  t_vector const coefficients;

  //! Holds filters for direct transform
  struct DirectFilter {
    //! Low-pass filter for direct transform
    t_vector low;
    //! High-pass filter for direct transform
    t_vector high;
  } const direct_filter;

  //! Holds filters for indirect transform
  struct {
    //! High-pass filter for direct transform
    t_vector low_even;
    t_vector low_odd;
    t_vector high_even;
    t_vector high_odd;
  } const indirect_filter;

  //! Constructs from initializers
  WaveletData(std::initializer_list<t_scalar> const &coefs);
  //! Constructs from vector
  WaveletData(t_vector const &coefs);
};

//! Factory function returning specific daubechie wavelet data
WaveletData const &daubechies_data(t_uint);
}
}
#endif
