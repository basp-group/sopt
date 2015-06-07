#include <type_traits>
#include "traits.h"

// Function inside anonymouns namespace won't appear in library
namespace sopt { namespace wavelets { namespace {
//! \brief Returns evaluated expression or copy of input
//! \details Gets C++ to figure out what the exact type is. Eigen tries and avoids copies. But
//! sometimes we actually want a copy, to make sure the arguments of a function are not modified
template<class T0>
  auto copy(Eigen::MatrixBase<T0> const &a)
  -> typename std::remove_const<typename std::remove_reference<decltype(a.eval())>::type>::type {
    return a.eval();
  }

//! \brief Applies scalar product of the size b
//! \details In practice, the operation is equivalent to the following
//! - `a` can be seen as a periodic vector: `a[i] == a[i % a.size()]`
//! - `a'` is the segment `a` = a[offset:offset+b.size()]`
//! - the result is the scalar product of `a'` by `b`.
template<class T0, class T2>
  typename T0::Scalar periodic_scalar_product(
      Eigen::MatrixBase<T0> const &a, Eigen::MatrixBase<T2> const &b, t_int offset) {
    auto const Na = static_cast<t_int>(a.size());
    auto const Nb = static_cast<t_int>(b.size());
    // offset in [0, a.size()[
    offset %= Na;
    if(offset < 0) offset += Na;
    // Simple case, just do it
    if(Na > Nb + offset)
      return (a.segment(offset, Nb).transpose() * b);
    // Wrap around, do it, but carefully
    typename T0::Scalar result(0);
    for(t_int i(0), j(offset); i < Nb; ++i, ++j)
      result += a(j % Na) * b(i);
    return result;
  }

//! \brief Convolves the signal by the filter
//! \details The signal is seen as a periodic vector during convolution
template<class T0, class T1, class T2>
  void convolve(
      Eigen::MatrixBase<T0> &result,
      Eigen::MatrixBase<T1> const &signal,
      Eigen::MatrixBase<T2> const &filter)
{
  assert(result.size() == signal.size());
  for(t_int i(0); i < t_int(signal.size()); ++i)
    result(i) = periodic_scalar_product(signal, filter, i);
}
//! \brief Convolve variation for vector blocks
template<class T0, class T1, class T2>
  void convolve(
      Eigen::VectorBlock<T0> &&result,
      Eigen::MatrixBase<T1> const &signal,
      Eigen::MatrixBase<T2> const &filter) {
    return convolve(result, signal, filter);
}


//! \brief Convolves and down-samples the signal by the filter
//! \details Just like convolve, but does every other point.
template<class T0, class T1, class T2>
  void down_convolve(
      Eigen::MatrixBase<T0> &result,
      Eigen::MatrixBase<T1> const &signal,
      Eigen::MatrixBase<T2> const &filter)
{
  assert(result.size() * 2 <= signal.size());
  for(t_int i(0); i < t_int(result.size()); ++i)
    result(i) = periodic_scalar_product(signal, filter, 2 * i);
}
//! \brief Dowsampling + convolve variation for vector blocks
template<class T0, class T1, class T2>
  void down_convolve(
      Eigen::VectorBlock<T0> &&result,
      Eigen::MatrixBase<T1> const &signal,
      Eigen::MatrixBase<T2> const &filter)
{
  down_convolve(result, signal, filter);
}

//! Convolve and sims low and high pass of a signal
template<class T0, class T1, class T2, class T3, class T4>
  void convolve_sum(
      Eigen::MatrixBase<T0> &result,
      Eigen::MatrixBase<T1> const &low_pass_signal,
      Eigen::MatrixBase<T2> const &low_pass,
      Eigen::MatrixBase<T3> const &high_pass_signal,
      Eigen::MatrixBase<T4> const &high_pass)
{
  assert(result.size() == low_pass_signal.size());
  assert(result.size() == high_pass_signal.size());
  assert(low_pass.size() == high_pass.size());

  auto const loffset = low_pass.size() - 1;
  auto const hoffset = high_pass.size() - 1;
  for(t_int i(0); i < static_cast<t_int>(result.size()); ++i)
    result(i) =
      periodic_scalar_product(low_pass_signal, low_pass, i - loffset)
      + periodic_scalar_product(high_pass_signal, high_pass, i - hoffset);
}

template<class WAVELET, class T0, class T1>
  typename std::enable_if<T1::IsVectorAtCompileTime, void>::type
  transform_impl(
      Eigen::MatrixBase<T0> &coeffs,
      Eigen::MatrixBase<T1> const& signal, WAVELET const &wavelet
  ) {
    assert(coeffs.rows() == signal.rows());
    assert(coeffs.cols() == signal.cols());
    assert(wavelet.direct_filter.low.size() == wavelet.direct_filter.high.size());

    auto const N = signal.size() >> 1;
    down_convolve(std::move(coeffs.head(N)), signal, wavelet.direct_filter.low);
    down_convolve(std::move(coeffs.tail(coeffs.size() - N)), signal, wavelet.direct_filter.high);
  }
template<class WAVELET, class T0, class T1>
  void transform_impl(
      Eigen::VectorBlock<T0> &&coeffs,
      Eigen::MatrixBase<T1> const& signal, WAVELET const &wavelet
  ) {
    transform_impl(coeffs, signal, wavelet);
  }

template<class WAVELET, class T0, class T1>
  typename std::enable_if<not T1::IsVectorAtCompileTime, void>::type
  transform_impl(
      Eigen::MatrixBase<T0> &coeffs,
      Eigen::MatrixBase<T1> &signal, WAVELET const &wavelet
  ) {
    assert(coeffs.rows() == signal.rows());
    assert(coeffs.cols() == signal.cols());
    assert(wavelet.direct_filter.low.size() == wavelet.direct_filter.high.size());

    auto const N = signal.size() >> 1;
    for(t_uint i(0); i < coeffs.rows(); ++i)
      transform(coeffs.row(i), signal.row(i), wavelet);

    for(t_uint i(0); i < coeffs.cols(); ++i) {
      signal.col(i) = coeffs.col(i);
      transform(coeffs.col(i), signal.col(i), wavelet);
    }
  }

template<class WAVELET, class T0, class T1>
  typename std::enable_if<T1::IsVectorAtCompileTime, void>::type
  transform(
      Eigen::MatrixBase<T0> &coeffs,
      Eigen::MatrixBase<T1> const& signal,
      t_uint levels, WAVELET const &wavelet
  ) {
    assert(coeffs.rows() == signal.rows());
    assert(coeffs.cols() == signal.cols());

    auto input = copy(signal);
    if(levels > 0)
      transform_impl(coeffs, input, wavelet);
    for(t_uint level(1); level < levels; ++level) {
      auto const N = signal.size() >> level;
      input.head(N) = coeffs.head(N);
      transform_impl(coeffs.head(N), input.head(N), wavelet);
    }
  }

template<class WAVELET, class T0, class T1>
  typename std::enable_if<not T1::IsVectorAtCompileTime, void>::type
  transform(
      Eigen::MatrixBase<T0> &coeffs,
      Eigen::MatrixBase<T1> const& signal,
      t_uint levels, WAVELET const& wavelet
  ) {
    assert(coeffs.rows() == signal.rows());
    assert(coeffs.cols() == signal.cols());

    auto input = copy(signal);
    if(levels > 0)
      transform_impl(coeffs, input, wavelet);
    for(t_uint level(1); level < levels; ++level) {
      auto const Nx = signal.rows() >> level;
      auto const Ny = signal.cols() >> level;
      input.block(Nx, Ny) = coeffs.head(Nx, Ny);
      transform_impl(coeffs.block(Nx, Ny), input.block(Nx, Ny), wavelet);
    }
  }

template<class WAVELET, class T0>
  auto transform(Eigen::MatrixBase<T0> const &signal, t_uint levels, WAVELET const& wavelet)
  -> decltype(copy(signal)) {
    auto result = copy(signal);
    transform(result, signal, levels, wavelet);
    return result;
  }

}
}}
