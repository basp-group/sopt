#include <type_traits>
#include <iostream>
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

  auto const loffset = 1 - static_cast<t_int>(low_pass.size());
  auto const hoffset = 1 - static_cast<t_int>(high_pass.size());
  for(t_int i(0); i < static_cast<t_int>(result.size()); ++i) {
      // std::cout << "F: " << low_pass_signal.transpose() << " x " << low_pass.transpose() << " for " << i + loffset << "\n";
    result(i) =
      periodic_scalar_product(low_pass_signal, low_pass, i + loffset)
      + periodic_scalar_product(high_pass_signal, high_pass, i + hoffset);
}
}

//! \brief Convolves and up-samples at the same time
//! \details If Cn shuffles the (periodic) vector right, U does the upsampling,
//! and O is the convolution operation, then the normal upsample + convolution
//! is O(Cn[U [a]], b). What we are doing here is U[O(Cn'[a'], b')]. This avoids
//! some unnecessary operations (multiplying and summing zeros) and removes the
//! need for temporary copies.  Testing shows the operations are equivalent. But
//! I certainly cannot show how on paper.
template<class T0, class T1, class T2, class T3, class T4, class T5>
  void up_convolve_sum(
      Eigen::MatrixBase<T0> &result,
      Eigen::MatrixBase<T1> const &coeffs,
      Eigen::MatrixBase<T2> const &low_even,
      Eigen::MatrixBase<T3> const &low_odd,
      Eigen::MatrixBase<T4> const &high_even,
      Eigen::MatrixBase<T5> const &high_odd)
{
  assert(result.size() % 2 == 0);
  assert(result.size() <= coeffs.size());
  assert(low_even.size() == high_even.size());
  assert(low_odd.size() == high_odd.size());
  auto const Nlow = (coeffs.size() + 1) >> 1;
  auto const Nhigh = coeffs.size() >> 1;
  auto const size = static_cast<t_int>(low_even.size() + low_odd.size());
  auto const is_even = size % 2 == 0;
  auto const even_offset = ((1 - size) >> 1) + (is_even ? 1: 0);
  auto const odd_offset = ((2 - size) >> 1) + (is_even ? 0: 1);
  for(t_int i(0); i+1 < static_cast<t_int>(result.size()); i += 2) {
      result(i + (is_even ? 1: 0)) =
            periodic_scalar_product(coeffs.head(Nlow), low_even, i/2 + even_offset)
            + periodic_scalar_product(coeffs.tail(Nhigh), high_even, i/2 + even_offset);
      result(i + (is_even ? 0: 1)) =
            periodic_scalar_product(coeffs.head(Nlow), low_odd, i/2 + odd_offset)
            + periodic_scalar_product(coeffs.tail(Nhigh), high_odd, i/2 + odd_offset);
  }
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
