#ifndef SOPT_WAVELETS_INNARDS_H
#define SOPT_WAVELETS_INNARDS_H

#include <Eigen/Core>
#include <iostream>

// Function inside anonymouns namespace won't appear in library
namespace sopt {
namespace wavelets {
namespace {
//! \brief Returns evaluated expression or copy of input
//! \details Gets C++ to figure out what the exact type is. Eigen tries and avoids copies. But
//! sometimes we actually want a copy, to make sure the arguments of a function are not modified
template <class T0>
auto copy(Eigen::ArrayBase<T0> const &a) ->
    typename std::remove_const<typename std::remove_reference<decltype(a.eval())>::type>::type {
  return a.eval();
}

//! \brief Applies scalar product of the size b
//! \details In practice, the operation is equivalent to the following
//! - `a` can be seen as a periodic vector: `a[i] == a[i % a.size()]`
//! - `a'` is the segment `a` = a[offset:offset+b.size()]`
//! - the result is the scalar product of `a'` by `b`.
template <class T0, class T2>
typename T0::Scalar
periodic_scalar_product(Eigen::ArrayBase<T0> const &a, Eigen::ArrayBase<T2> const &b,
                        typename T0::Index offset) {
  auto const Na = static_cast<typename T0::Index>(a.size());
  auto const Nb = static_cast<typename T0::Index>(b.size());
  // offset in [0, a.size()[
  offset %= Na;
  if(offset < 0)
    offset += Na;
  // Simple case, just do it
  if(Na > Nb + offset) {
    return (a.segment(offset, Nb) * b).sum();
  }
  // Wrap around, do it, but carefully
  typename T0::Scalar result(0);
  for(typename T0::Index i(0), j(offset); i < Nb; ++i, ++j)
    result += a(j % Na) * b(i);
  return result;
}

//! \brief Convolves the signal by the filter
//! \details The signal is seen as a periodic vector during convolution
template <class T0, class T1, class T2>
void convolve(Eigen::ArrayBase<T0> &result, Eigen::ArrayBase<T1> const &signal,
              Eigen::ArrayBase<T2> const &filter) {
  assert(result.size() == signal.size());
  for(typename T0::Index i(0); i < t_int(signal.size()); ++i)
    result(i) = periodic_scalar_product(signal, filter, i);
}
//! \brief Convolve variation for vector blocks
template <class T0, class T1, class T2>
void convolve(Eigen::VectorBlock<T0> &&result, Eigen::ArrayBase<T1> const &signal,
              Eigen::ArrayBase<T2> const &filter) {
  return convolve(result, signal, filter);
}

//! \brief Convolves and down-samples the signal by the filter
//! \details Just like convolve, but does every other point.
template <class T0, class T1, class T2>
void down_convolve(Eigen::ArrayBase<T0> &result, Eigen::ArrayBase<T1> const &signal,
                   Eigen::ArrayBase<T2> const &filter) {
  assert(result.size() * 2 <= signal.size());
  if(signal.rows() == 1)
    for(typename T0::Index i(0); i < result.size(); ++i)
      result(i) = periodic_scalar_product(signal.transpose(), filter, 2 * i);
  else
    for(typename T0::Index i(0); i < result.size(); ++i)
      result(i) = periodic_scalar_product(signal, filter, 2 * i);
}
//! \brief Dowsampling + convolve variation for vector blocks
template <class T0, class T1, class T2>
void down_convolve(Eigen::VectorBlock<T0> &&result, Eigen::ArrayBase<T1> const &signal,
                   Eigen::ArrayBase<T2> const &filter) {
  down_convolve(result, signal, filter);
}

//! Convolve and sims low and high pass of a signal
template <class T0, class T1, class T2, class T3, class T4>
void convolve_sum(Eigen::ArrayBase<T0> &result, Eigen::ArrayBase<T1> const &low_pass_signal,
                  Eigen::ArrayBase<T2> const &low_pass,
                  Eigen::ArrayBase<T3> const &high_pass_signal,
                  Eigen::ArrayBase<T4> const &high_pass) {
  assert(result.size() == low_pass_signal.size());
  assert(result.size() == high_pass_signal.size());
  assert(low_pass.size() == high_pass.size());
  static_assert(std::is_signed<typename T0::Index>::value, "loffset, hoffset expect signed values");
  auto const loffset = 1 - static_cast<typename T0::Index>(low_pass.size());
  auto const hoffset = 1 - static_cast<typename T0::Index>(high_pass.size());
  for(typename T0::Index i(0); i < result.size(); ++i) {
    result(i) = periodic_scalar_product(low_pass_signal, low_pass, i + loffset)
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
template <class T0, class T1, class T2, class T3, class T4, class T5>
void up_convolve_sum(Eigen::ArrayBase<T0> &result, Eigen::ArrayBase<T1> const &coeffs,
                     Eigen::ArrayBase<T2> const &low_even, Eigen::ArrayBase<T3> const &low_odd,
                     Eigen::ArrayBase<T4> const &high_even, Eigen::ArrayBase<T5> const &high_odd) {
  assert(result.size() <= coeffs.size());
  assert(result.size() % 2 == 0);
  assert(low_even.size() == high_even.size());
  assert(low_odd.size() == high_odd.size());
  auto const Nlow = (coeffs.size() + 1) / 2;
  auto const Nhigh = coeffs.size() / 2;
  auto const size = low_even.size() + low_odd.size();
  auto const is_even = size % 2 == 0;
  auto const even_offset = (1 - size) / 2;
  auto const odd_offset = (1 - size) / 2 + (is_even ? 0 : 1);
  for(typename T0::Index i(0); i + 1 < result.size(); i += 2) {
    result(i + (is_even ? 1 : 0))
        = periodic_scalar_product(coeffs.head(Nlow), low_even, i / 2 + even_offset)
          + periodic_scalar_product(coeffs.tail(Nhigh), high_even, i / 2 + even_offset);
    result(i + (is_even ? 0 : 1))
        = periodic_scalar_product(coeffs.head(Nlow), low_odd, i / 2 + odd_offset)
          + periodic_scalar_product(coeffs.tail(Nhigh), high_odd, i / 2 + odd_offset);
  }
}
}
}
}
#endif
