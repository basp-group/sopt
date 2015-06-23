#ifndef SOPT_UTILITY_H
#define SOPT_UTILITY_H

#include <Eigen/Core>
#include <type_traits>
#include <algorithm>
#include <complex>

namespace sopt {

namespace details {
  //! Expression to create projection onto positive orthant
  template<class SCALAR> class ProjectPositiveQuadrant {
     public:
       SCALAR operator()(const SCALAR &value) const { return std::max(value, SCALAR(0)); }
  };
  //! Specialization for complex numbers
  template<class SCALAR> class ProjectPositiveQuadrant<std::complex<SCALAR>> {
     public:
       SCALAR operator()(SCALAR const &value) const {
         return std::max(value, SCALAR(0));
       }
       std::complex<SCALAR> operator()(std::complex<SCALAR> const &value) const {
         return std::complex<SCALAR>((*this)(value.real()), SCALAR(0));
       }
  };
}

//! Expression to create projection onto positive quadrant
template<class T>
Eigen::CwiseUnaryOp<details::ProjectPositiveQuadrant<typename T::Scalar>, const T>
positive_quadrant(Eigen::DenseBase<T> const &input) {
  typedef details::ProjectPositiveQuadrant<typename T::Scalar> Projector;
  typedef Eigen::CwiseUnaryOp<Projector, const T> UnaryOp;
  return UnaryOp(input.derived(), Projector());
}

} /* sopt */ 
#endif
