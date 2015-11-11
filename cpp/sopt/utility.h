#ifndef SOPT_UTILITY_H
#define SOPT_UTILITY_H

#include <Eigen/Core>
#include <type_traits>
#include <algorithm>
#include <complex>

#include "sopt/types.h"

namespace sopt {

//! abs(x) < threshhold ? 0: x - sgn(x) * threshhold
template<class SCALAR>
  SCALAR soft_threshhold(SCALAR const & x, typename real_type<SCALAR>::type const & threshhold) {
    auto const normalized = std::abs(x);
    return normalized < threshhold ? SCALAR(0): (x * (SCALAR(1) - threshhold/normalized));
  }

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

  //! Expression to create projection onto positive orthant
  template<class SCALAR>
    using SoftThreshhold = decltype(
        std::bind(
          soft_threshhold<SCALAR>,
          std::placeholders::_1,
          typename real_type<SCALAR>::type(1)
        )
    );
}

//! Expression to create projection onto positive quadrant
template<class T>
  Eigen::CwiseUnaryOp<const details::ProjectPositiveQuadrant<typename T::Scalar>, const T>
  positive_quadrant(Eigen::DenseBase<T> const &input) {
    typedef details::ProjectPositiveQuadrant<typename T::Scalar> Projector;
    typedef Eigen::CwiseUnaryOp<const Projector, const T> UnaryOp;
    return UnaryOp(input.derived(), Projector());
  }

//! Expression to create soft-threshhold
template<class T>
  Eigen::CwiseUnaryOp<const details::SoftThreshhold<typename T::Scalar>, const T>
  soft_threshhold(Eigen::DenseBase<T> const &input, typename T::Scalar const &threshhold) {
    typedef typename T::Scalar Scalar;
    typedef typename real_type<Scalar>::type Real;
    return {
      input.derived(),
      std::bind(soft_threshhold<Scalar>, std::placeholders::_1, Real(threshhold))
    };
  }

//! Computes weighted L1 norm
template<class T0, class T1>
  typename real_type<typename T0::Scalar>::type l1_norm(
    Eigen::ArrayBase<T0> const& input, Eigen::ArrayBase<T1> const &weight) {
      return (input.cwiseAbs() * weight).real().sum();
    }
//! Computes weighted L1 norm
template<class T0, class T1>
  typename real_type<typename T0::Scalar>::type l1_norm(
    Eigen::MatrixBase<T0> const& input, Eigen::MatrixBase<T1> const &weight) {
      return l1_norm(input.array(), weight.array());
    }
//! Computes L1 norm
template<class T0>
  typename real_type<typename T0::Scalar>::type l1_norm(
    Eigen::ArrayBase<T0> const& input) { return input.cwiseAbs().sum(); }
//! Computes L1 norm
template<class T0>
  typename real_type<typename T0::Scalar>::type l1_norm(
    Eigen::MatrixBase<T0> const& input) { return l1_norm(input.array()); }

namespace details {
  //! Greatest common divisor
  inline t_int gcd(t_int a, t_int b) { return b == 0 ? a : gcd(b, a % b); }
}


} /* sopt */
#endif
