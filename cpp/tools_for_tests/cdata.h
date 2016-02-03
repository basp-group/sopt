#ifndef SOPT_TOOLS_FOR_TESTS_CDATA
#define SOPT_TOOLS_FOR_TESTS_CDATA
#include "sopt/linear_transform.h"
#include <Eigen/Core>

namespace sopt {
// Wraps calls to sampling and wavelets to C style
template <class T> struct CData {
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> t_Vector;
  typename t_Vector::Index nin, nout;
  sopt::LinearTransform<t_Vector> const &transform;
};

template <class T> void direct_transform(void *out, void *in, void **data) {
  CData<T> const &cdata = *(CData<T> *)data;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> t_Vector;
  t_Vector const eval = cdata.transform * t_Vector::Map((T *)in, cdata.nin);
  t_Vector::Map((T *)out, cdata.nout) = eval;
}
template <class T> void adjoint_transform(void *out, void *in, void **data) {
  CData<T> const &cdata = *(CData<T> *)data;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> t_Vector;
  t_Vector const eval = cdata.transform.adjoint() * t_Vector::Map((T *)in, cdata.nout);
  t_Vector::Map((T *)out, cdata.nin) = eval;
}

} /* sopt */
#endif
