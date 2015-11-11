#ifndef SOPT_UTILITY_H
#define SOPT_UTILITY_H

#include <Eigen/Core>
#include <type_traits>
#include <algorithm>
#include <complex>

#include "sopt/types.h"
#include "sopt/exception.h"

namespace sopt {

//! abs(x) < threshhold ? 0: x - sgn(x) * threshhold
template<class SCALAR>
  typename std::enable_if<
    // standard_layout allows SCALAR = std::complex
    // and disallows Eigen::EigenBase<DERIVED> objects.
    // Also allows stuff it shouldn't... but whatever.
    std::is_arithmetic<SCALAR>::value or std::is_pod<SCALAR>::value,
    SCALAR
  >::type soft_threshhold(SCALAR const & x, typename real_type<SCALAR>::type const & threshhold) {
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

  //! Helper template typedef to instantiate soft_threshhold that takes an Eigen object
  template<class SCALAR>
    using SoftThreshhold = decltype(
        std::bind(
          soft_threshhold<SCALAR>,
          std::placeholders::_1,
          typename real_type<SCALAR>::type(1)
        )
    );

  //! Helper template typedef defining binary operation
  //! Merely defines a pointer to the right kind of function for eigen binary op.
  template<class SCALAR>
    using BinaryOp = SCALAR(*)(SCALAR const&, SCALAR const&);
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

//! \brief Expression to create soft-threshhold with multiple parameters
//! \details Operates over a vector of threshholds: ``out(i) = soft_threshhold(x(i), h(i))``
//! Threshhold and input vectors must have the same size.
template<class T0, class T1>
  Eigen::CwiseBinaryOp<details::BinaryOp<typename T0::Scalar>, const T0, const T1>
  soft_threshhold(Eigen::DenseBase<T0> const &input, Eigen::DenseBase<T1> const &threshhold) {
    if(input.size() != threshhold.size())
      SOPT_THROW("Threshhold and input should have the same size");
    return {input.derived(), threshhold.derived(), soft_threshhold<typename T0::Scalar>};
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
