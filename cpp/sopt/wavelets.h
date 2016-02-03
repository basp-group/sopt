#ifndef SOPT_WAVELETS_H
#define SOPT_WAVELETS_H

// Convenience header to include wavelets headers and additional utilities

#include "sopt/config.h"
#include "sopt/linear_transform.h"
#include "wavelets/sara.h"
#include "wavelets/wavelets.h"

namespace sopt {
//! Thin linear-transform wrapper around 1d wavelets
template <class T> LinearTransform<Vector<T>> linear_transform(wavelets::Wavelet const &wavelet) {
  return LinearTransform<Vector<T>>(
      [&wavelet](Vector<T> &out, Vector<T> const &x) { wavelet.direct(out.array(), x.array()); },
      [&wavelet](Vector<T> &out, Vector<T> const &x) { wavelet.indirect(x.array(), out.array()); });
}

//! Thin linear-transform wrapper around 2d wavelets
template <class T>
LinearTransform<Vector<T>>
linear_transform(wavelets::Wavelet const &wavelet, t_uint rows, t_uint cols = 1) {
  if(rows == 1 or cols == 1)
    return linear_transform<T>(wavelet);
  return LinearTransform<Vector<T>>(
      [&wavelet, rows, cols](Vector<T> &out, Vector<T> const &x) {
        assert(x.size() == rows * cols);
        out.resize(x.size());
        auto const x_mat = Image<T>::Map(x.data(), rows, cols);
        auto out_mat = Image<T>::Map(out.data(), rows, cols);
        wavelet.indirect(x_mat, out_mat);
      },
      [&wavelet, rows, cols](Vector<T> &out, Vector<T> const &x) {
        assert(x.size() == rows * cols);
        out.resize(x.size());
        auto const x_mat = Image<T>::Map(x.data(), rows, cols);
        auto out_mat = Image<T>::Map(out.data(), rows, cols);
        wavelet.direct(out_mat, x_mat);
      });
}

//! Thin linear-transform wrapper around 1d sara operator
template <class T> LinearTransform<Vector<T>> linear_transform(wavelets::SARA const &wavelet) {
  return LinearTransform<Vector<T>>(
      [wavelet](Vector<T> &out, Vector<T> const &x) { wavelet.direct(out, x); },
      {{wavelet.size(), 1, 0}},
      [wavelet](Vector<T> &out, Vector<T> const &x) { wavelet.indirect(out, x); },
      {{1, wavelet.size(), 0}});
}
} /* sopt */

#endif
