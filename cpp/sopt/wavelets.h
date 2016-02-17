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
      [&op](Vector<T> &out, Vector<T> const &x) { op.indirect(x.array(), out.array()); },
      [&op](Vector<T> &out, Vector<T> const &x) { op.direct(out.array(), x.array()); });
}
//! \brief Thin linear-transform wrapper around 2d wavelets
//! \details Goes back and forth between vector representations and image representations, where
//! images can be 2d.
//! \param[in] op: Wavelet operator
//! \param[in] rows: Number of rows in the image
//! \param[in] cols: Number of columns in the image
//! \param[in] factor: Allows for SARA transforms, i.e. more than one wavelet basis
template <class T, class OP>
LinearTransform<Vector<T>>
linear_transform(OP const &op, t_uint rows, t_uint cols, t_uint factor = 1) {
  if(rows == 1 or cols == 1)
    return linear_transform<T, OP>(op);
  return LinearTransform<Vector<T>>(
      [&op, rows, cols, factor](Vector<T> &out, Vector<T> const &x) {
        assert(static_cast<t_uint>(x.size()) == rows * cols * factor);
        out.resize(rows * cols);
        auto signal = Image<T>::Map(out.data(), rows, cols);
        auto const coeffs = Image<T>::Map(x.data(), rows, cols * factor);
        op.indirect(coeffs, signal);
      },
      {0, 1, static_cast<t_int>(rows * cols)},
      [&op, rows, cols, factor](Vector<T> &out, Vector<T> const &x) {
        assert(static_cast<t_uint>(x.size()) == rows * cols);
        out.resize(rows * cols * factor);
        auto const signal = Image<T>::Map(x.data(), rows, cols);
        auto coeffs = Image<T>::Map(out.data(), rows, cols * factor);
        op.direct(coeffs, signal);
      },
      {0, 1, static_cast<t_int>(factor * rows * cols)});
}
} // anonymous
} // details

//! \brief Thin linear-transform wrapper around 1d wavelets
//! \warning Because of the way Purify defines things, Ψ^T is actually the transform from signal to
//! coefficients.
template <class T> LinearTransform<Vector<T>> linear_transform(wavelets::Wavelet const &wavelet) {
  return details::linear_transform<T, wavelets::Wavelet>(wavelet);
}

//! \brief Thin linear-transform wrapper around 1d sara operator
//! \note Because of the way Purify defines things, Ψ^T is actually the transform from signal to
//! coefficients.
template <class T> LinearTransform<Vector<T>> linear_transform(wavelets::SARA const &sara) {
  return details::linear_transform<T, wavelets::SARA>(sara);
}

//! \brief Thin linear-transform wrapper around 2d wavelets
//! \note Because of the way Purify defines things, Ψ^T is actually the transform from signal to
//! coefficients.
template <class T>
LinearTransform<Vector<T>>
linear_transform(wavelets::Wavelet const &wavelet, t_uint rows, t_uint cols = 1) {
  return details::linear_transform<T, wavelets::Wavelet>(wavelet, rows, cols);
}
//! \brief Thin linear-transform wrapper around 2d wavelets
//! \param[in] sara: SARA wavelet dictionary
//! \param[in] rows: Number of rows in the image
//! \param[in] cols: Number of columns in the image
//! \note Because of the way Purify defines things, Ψ^T is actually the transform from signal to
//! coefficients.
template <class T>
LinearTransform<Vector<T>>
linear_transform(wavelets::SARA const &sara, t_uint rows, t_uint cols = 1) {
  return details::linear_transform<T, wavelets::SARA>(sara, rows, cols, sara.size());
}
} /* sopt */

#endif
