#ifndef SOPT_WAVELETS_H
#define SOPT_WAVELETS_H

// Convenience header to include wavelets headers and additional utilities

#include "wavelets/wavelets.h"
#include "wavelets/sara.h"
#include "sopt/linear_transform.h"

namespace sopt {
  //! Thin linear-transform wrapper around 1d wavelets
  template<class T>
  LinearTransform<Eigen::Matrix<T, Eigen::Dynamic, 1>>
  linear_transform(wavelets::Wavelet const &wavelet) {
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> t_Vector;
    return LinearTransform<t_Vector>(
       [&wavelet](t_Vector &out, t_Vector const &x) { wavelet.direct(out.array(), x.array()); },
       [&wavelet](t_Vector &out, t_Vector const &x) { wavelet.indirect(x.array(), out.array()); }
    );
  }

  //! Thin linear-transform wrapper around 2d wavelets
  template<class T>
  LinearTransform<Eigen::Matrix<T, Eigen::Dynamic, 1>>
  linear_transform(wavelets::Wavelet const &wavelet, t_uint rows, t_uint cols = 1) {
    if(rows == 1 or cols == 1)
      return linear_transform<T>(wavelet);
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> t_Vector;
    return LinearTransform<t_Vector>(
       [&wavelet, rows, cols](t_Vector &out, t_Vector const &x) {
         assert(x.size() == rows * cols);
         out.resize(x.size());
         typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> t_Array;
         auto const x_mat = t_Array::Map(x.data(), rows, cols);
         auto out_mat = t_Array::Map(out.data(), rows, cols);
         wavelet.indirect(x_mat, out_mat);
       },
       [&wavelet, rows, cols](t_Vector &out, t_Vector const &x) {
         assert(x.size() == rows * cols);
         out.resize(x.size());
         typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> t_Array;
         auto const x_mat = t_Array::Map(x.data(), rows, cols);
         auto out_mat = t_Array::Map(out.data(), rows, cols);
         wavelet.direct(out_mat, x_mat);
       }
    );
  }

  //! Thin linear-transform wrapper around 1d sara operator
  template<class T>
  LinearTransform<Eigen::Matrix<T, Eigen::Dynamic, 1>>
  linear_transform(wavelets::SARA const &wavelet) {
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> t_Vector;
    return LinearTransform<t_Vector>(
       [wavelet](t_Vector &out, t_Vector const &x) { wavelet.direct(out, x); },
       {{wavelet.size(), 1, 0}},
       [wavelet](t_Vector &out, t_Vector const &x) { wavelet.indirect(out, x); },
       {{1, wavelet.size(), 0}}
    );
  }
} /* sopt */

#endif
