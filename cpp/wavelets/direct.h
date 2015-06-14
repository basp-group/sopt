#ifndef SOPT_WAVELETS_DIRECT_H
#define SOPT_WAVELETS_DIRECT_H
#include <type_traits>
#include "types.h"

// Function inside anonymouns namespace won't appear in library
namespace sopt { namespace wavelets {

namespace {
  //! \brief Single-level 1d direct transform
  //! \param[out] coeffs_: output of the function (despite the const)
  //! \param[in] signal: input signal for which to compute wavelet transform
  //! \param[in] wavelet: contains wavelet coefficients
  template<class WAVELET, class T0, class T1>
    typename std::enable_if<T1::IsVectorAtCompileTime, void>::type
    direct_transform_impl(
        Eigen::MatrixBase<T0> const& coeffs_,
        Eigen::MatrixBase<T1> const& signal, WAVELET const &wavelet
    ) {
      Eigen::MatrixBase<T0> & coeffs = const_cast< Eigen::MatrixBase<T0>& >(coeffs_);
      assert(coeffs.size() == signal.size());
      assert(wavelet.direct_filter.low.size() == wavelet.direct_filter.high.size());

      auto const N = signal.size() / 2;
      down_convolve(coeffs.head(N), signal, wavelet.direct_filter.low);
      down_convolve(coeffs.tail(coeffs.size() - N), signal, wavelet.direct_filter.high);
    }

  //! Single-level 2d direct transform
  //! \param[out] coeffs_: output of the function (despite the const)
  //! \param[inout] signal: input signal for which to compute wavelet transform. Input is modified.
  //! \param[in] wavelet: contains wavelet coefficients
  template<class WAVELET, class T0, class T1>
    typename std::enable_if<not T1::IsVectorAtCompileTime, void>::type
    direct_transform_impl(
        Eigen::MatrixBase<T0> const &coeffs_,
        Eigen::MatrixBase<T1> const &signal_, WAVELET const &wavelet
    ) {
      Eigen::MatrixBase<T0> & coeffs = const_cast< Eigen::MatrixBase<T0>& >(coeffs_);
      Eigen::MatrixBase<T0> & signal = const_cast< Eigen::MatrixBase<T0>& >(signal_);
      assert(coeffs.rows() == signal.rows());
      assert(coeffs.cols() == signal.cols());
      assert(wavelet.direct_filter.low.size() == wavelet.direct_filter.high.size());

      auto const N = signal.size() / 2;
      for(t_uint i(0); i < coeffs.rows(); ++i)
        direct_transform_impl(coeffs.row(i).transpose(), signal.row(i).transpose(), wavelet);

      for(t_uint i(0); i < coeffs.cols(); ++i) {
        signal.col(i) = coeffs.col(i);
        direct_transform_impl(coeffs.col(i), signal.col(i), wavelet);
      }
    }
}

//! N-levels 1d direct transform
template<class WAVELET, class T0, class T1>
  typename std::enable_if<T1::IsVectorAtCompileTime, void>::type
  direct_transform(
      Eigen::MatrixBase<T0> &coeffs,
      Eigen::MatrixBase<T1> const& signal,
      t_uint levels, WAVELET const &wavelet
  ) {
    assert(coeffs.rows() == signal.rows());
    assert(coeffs.cols() == signal.cols());

    auto input = copy(signal);
    if(levels > 0)
      direct_transform_impl(coeffs, input, wavelet);
    for(t_uint level(1); level < levels; ++level) {
      auto const N = static_cast<t_uint>(signal.size()) >> level;
      input.head(N) = coeffs.head(N);
      direct_transform_impl(coeffs.head(N), input.head(N), wavelet);
    }
  }
//! N-levels 2d direct transform
template<class WAVELET, class T0, class T1>
  typename std::enable_if<not T1::IsVectorAtCompileTime, void>::type
  direct_transform(
      Eigen::MatrixBase<T0> &coeffs,
      Eigen::MatrixBase<T1> const& signal,
      t_uint levels, WAVELET const& wavelet
  ) {
    assert(coeffs.rows() == signal.rows());
    assert(coeffs.cols() == signal.cols());

    auto input = copy(signal);
    if(levels > 0)
      direct_transform_impl(coeffs, input, wavelet);
    for(t_uint level(1); level < levels; ++level) {
      auto const Nx = static_cast<t_uint>(signal.rows()) >> level;
      auto const Ny = static_cast<t_uint>(signal.cols()) >> level;
      input.topLeftCorner(Nx, Ny) = coeffs.topLeftCorner(Nx, Ny);
      direct_transform_impl(coeffs.topLeftCorner(Nx, Ny), input.topLeftCorner(Nx, Ny), wavelet);
    }
  }

//! Direct 1d and 2d transform
template<class WAVELET, class T0>
  auto direct_transform(Eigen::MatrixBase<T0> const &signal, t_uint levels, WAVELET const& wavelet)
  -> decltype(copy(signal)) {
    auto result = copy(signal);
    direct_transform(result, signal, levels, wavelet);
    return result;
  }
}}
#endif
