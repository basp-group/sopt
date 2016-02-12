#ifndef SOPT_WAVELETS_H
#define SOPT_WAVELETS_H

// Convenience header to include wavelets headers and additional utilities

#include "sopt/config.h"
#include "sopt/linear_transform.h"
#include "sopt/wavelets/sara.h"
#include "sopt/wavelets/wavelets.h"

namespace sopt {
namespace details {
namespace {
//! Thin linear-transform wrapper around some operator accepting direct and indirect
template <class T, class OP> LinearTransform<Vector<T>> linear_transform(OP const &op) {
  return LinearTransform<Vector<T>>(
      [&op](Vector<T> &out, Vector<T> const &x) { op.direct(out.array(), x.array()); },
      [&op](Vector<T> &out, Vector<T> const &x) { op.indirect(x.array(), out.array()); });
}
//! Thin linear-transform wrapper around 2d wavelets and some operator accepting direct and indirect
template <class T, class OP>
LinearTransform<Vector<T>> linear_transform(OP const &op, t_uint rows, t_uint cols) {
  if(rows == 1 or cols == 1)
    return linear_transform<T, OP>(op);
  return LinearTransform<Vector<T>>(
      [&op, rows, cols](Vector<T> &out, Vector<T> const &x) {
        assert(static_cast<t_uint>(x.size()) == rows * cols);
        out.resize(x.size());
        auto const x_mat = Image<T>::Map(x.data(), rows, cols);
        auto out_mat = Image<T>::Map(out.data(), rows, cols);
        op.indirect(x_mat, out_mat);
      },
      [&op, rows, cols](Vector<T> &out, Vector<T> const &x) {
        assert(static_cast<t_uint>(x.size()) == rows * cols);
        out.resize(x.size());
        auto const x_mat = Image<T>::Map(x.data(), rows, cols);
        auto out_mat = Image<T>::Map(out.data(), rows, cols);
        op.direct(out_mat, x_mat);
      });
}
} // anonymous
} // details

//! Thin linear-transform wrapper around 1d wavelets
template <class T> LinearTransform<Vector<T>> linear_transform(wavelets::Wavelet const &wavelet) {
  return details::linear_transform<T, wavelets::Wavelet>(wavelet);
}

//! Thin linear-transform wrapper around 1d sara operator
template <class T> LinearTransform<Vector<T>> linear_transform(wavelets::SARA const &sara) {
  return details::linear_transform<T, wavelets::SARA>(sara);
}

//! Thin linear-transform wrapper around 2d wavelets
template <class T>
LinearTransform<Vector<T>>
linear_transform(wavelets::Wavelet const &wavelet, t_uint rows, t_uint cols = 1) {
  return details::linear_transform<T, wavelets::Wavelet>(wavelet, rows, cols);
}
//! Thin linear-transform wrapper around 2d wavelets
template <class T>
LinearTransform<Vector<T>>
linear_transform(wavelets::SARA const &sara, t_uint rows, t_uint cols = 1) {
  return details::linear_transform<T, wavelets::SARA>(sara, rows, cols);
}
} /* sopt */

#endif
