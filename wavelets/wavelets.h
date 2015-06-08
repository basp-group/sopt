#ifndef SOPT_WAVELETS_WAVELETS_H
#define SOPT_WAVELETS_WAVELETS_H

#include <Eigen/Core>
#include "traits.h"

namespace sopt { namespace wavelets {

  //! Holds wavelets coefficients
  struct WaveletData {
    //! Type of the underlying scalar
    typedef t_real t_scalar;
    //! Type of the underlying vector
    typedef Eigen::Matrix<t_scalar, Eigen::Dynamic, 1> t_vector;
    //! Wavelet coefficient per-se
    const t_vector coefficients;

    //! Holds filters for direct transform
    struct {
      //! Low-pass filter for direct transform
      const t_vector low;
      //! High-pass filter for direct transform
      const t_vector high;
    } direct_filter;

    //! Holds filters for indirect transform
    struct {
      //! High-pass filter for direct transform
      const t_vector low_even;
      const t_vector low_odd;
      const t_vector high_even;
      const t_vector high_odd;
    } indirect_filter;

    //! Constructs from initializers
    WaveletData(std::initializer_list<t_scalar> const &coefs);
    //! Constructs from vector
    WaveletData(t_vector const &coefs);
  };

  extern const WaveletData Daubechies1;
  extern const WaveletData Daubechies2;
  extern const WaveletData Daubechies3;
  extern const WaveletData Daubechies4;
  extern const WaveletData Daubechies5;
  extern const WaveletData Daubechies6;
  extern const WaveletData Daubechies7;
  extern const WaveletData Daubechies8;
  extern const WaveletData Daubechies9;
  extern const WaveletData Daubechies10;
  extern const WaveletData Daubechies11;
  extern const WaveletData Daubechies12;
  extern const WaveletData Daubechies13;
  extern const WaveletData Daubechies14;
  extern const WaveletData Daubechies15;
  extern const WaveletData Daubechies16;
  extern const WaveletData Daubechies17;
  extern const WaveletData Daubechies18;
  extern const WaveletData Daubechies19;
  extern const WaveletData Daubechies20;
  extern const WaveletData Daubechies21;
  extern const WaveletData Daubechies22;
  extern const WaveletData Daubechies23;
  extern const WaveletData Daubechies24;
  extern const WaveletData Daubechies25;
  extern const WaveletData Daubechies26;
  extern const WaveletData Daubechies27;
  extern const WaveletData Daubechies28;
  extern const WaveletData Daubechies29;
  extern const WaveletData Daubechies30;
  extern const WaveletData Daubechies31;
  extern const WaveletData Daubechies32;
  extern const WaveletData Daubechies33;
  extern const WaveletData Daubechies34;
  extern const WaveletData Daubechies35;
  extern const WaveletData Daubechies36;
  extern const WaveletData Daubechies37;
  extern const WaveletData Daubechies38;

  //! Returns specific daubechie wavelet
  WaveletData const & daubechies(t_uint);
}}
#endif
