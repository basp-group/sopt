#ifndef SOPT_L1_PROXIMAL_H
#define SOPT_L1_PROXIMAL_H

#include <iostream>
#include <type_traits>
#include <Eigen/Core>

#include "sopt/utility.h"
#include "sopt/proximal_expression.h"
#include "sopt/linear_transform.h"

namespace sopt { namespace proximal {

//! \brief L1 proximal, including linear transform
//! \details This function computes the prox operator of the l1
//!  norm for the input vector \f$x\f$. It solves the problem:
//!  \f[ min_{z} 0.5||x - z||_2^2 + γ ||Ψ^† z||_w1 \f]
//!  where \f$Ψ \in C^{N_x \times N_r} \f$ is the sparsifying operator, and \f[|| ||_w1\f] is the
//!  weighted L1 norm.
template<class SCALAR> class L1TightFrame
{
  public:
    //! Underlying scalar type
    typedef SCALAR Scalar;
    //! Underlying real scalar type
    typedef typename real_type<Scalar>::type Real;

    L1TightFrame() : Psi_(linear_transform_identity<Scalar>()), nu_(1e0), weights_(1) {
      weights_(0) = 1;
    }

#   define SOPT_MACRO(NAME, TYPE)                                                           \
        TYPE const& NAME() const { return NAME ## _; }                                      \
        L1TightFrame<Scalar> & NAME(TYPE const &NAME) { NAME ## _ = NAME; return *this; }   \
      protected:                                                                            \
        TYPE NAME ## _;                                                                     \
      public:
    //! Linear transform applied to input prior to L1 norm
    SOPT_MACRO(Psi, LinearTransform< Vector<Scalar> >);
    //! Bound on the squared norm of the operator Psi
    SOPT_MACRO(nu, Real);
    //! Conjugate gradient
    SOPT_MACRO(weights, Vector<Real>);
#   undef SOPT_MACRO

    //! Set weights to a single value
    L1TightFrame<Scalar> & weights(Real const &value) {
      weights_ = Vector<Real>::Ones(1) * value;
      return *this;
    }

    //! Set Psi and Psi^† using a matrix
    template<class T>
      L1TightFrame& Psi(Eigen::MatrixBase<T> const &psi) {
        return Psi(linear_transform(psi));
      }

    //! \brief Computes proximal for given gamm
    //! \details Argument gamma temporaly changes the class member lambda.
    //! This is the signature proximal functions generally use.
    template<class T0, class T1>
    typename std::enable_if<
      is_complex<Scalar>::value == is_complex<typename T0::Scalar>::value
      and is_complex<Scalar>::value == is_complex<typename T1::Scalar>::value
    >::type operator()(
        Eigen::MatrixBase<T0> &out, Real gamma, Eigen::MatrixBase<T1> const &x) const;

    //! Lazy version
    template<class T0>
      ProximalExpression<L1TightFrame<Scalar> const&, T0>
      operator()(Real const &gamma, Eigen::MatrixBase<T0> const &x) const {
        return {*this, gamma, x};
      }

    //! \f[ 0.5||x - z||_2^2 + γ||Ψ^† z||_w1 \f]
    template<class T0, class T1>
    typename std::enable_if<
      is_complex<Scalar>::value == is_complex<typename T0::Scalar>::value
      and is_complex<Scalar>::value == is_complex<typename T1::Scalar>::value,
      Real
    > :: type objective(
        Eigen::MatrixBase<T0> const &x, Eigen::MatrixBase<T1> const &z, Real const &gamma) const;
};

template<class SCALAR> template<class T0, class T1>
typename std::enable_if<
  is_complex<SCALAR>::value == is_complex<typename T0::Scalar>::value
  and is_complex<SCALAR>::value == is_complex<typename T1::Scalar>::value
>::type L1TightFrame<SCALAR>::operator()(
    Eigen::MatrixBase<T0> &out, Real gamma, Eigen::MatrixBase<T1> const &x) const {

  Vector<Scalar> const psit_x = Psi().adjoint() * x;
  if(weights().size() == 1)
    out = Psi() * (soft_threshhold(psit_x, nu() * gamma * weights()(0)) - psit_x) / nu() + x;
  else
    out = Psi() * (soft_threshhold(psit_x, nu() * gamma * weights()) - psit_x) / nu() + x;
  SOPT_INFO("Prox L1: objective = {}", objective(x, out, gamma));
}

template<class SCALAR> template<class T0, class T1>
typename std::enable_if<
  is_complex<SCALAR>::value == is_complex<typename T0::Scalar>::value
  and is_complex<SCALAR>::value == is_complex<typename T1::Scalar>::value,
  typename real_type<SCALAR>::type
> :: type L1TightFrame<SCALAR>::objective(
    Eigen::MatrixBase<T0> const &x, Eigen::MatrixBase<T1> const &z, Real const &gamma) const {
  if(weights().size() == 1)
    return 0.5 * (x - z).squaredNorm()
      + gamma * sopt::l1_norm(Psi().adjoint() * z) * std::abs(weights()(0));
  else
    return 0.5 * (x - z).squaredNorm()
      + gamma * sopt::l1_norm(Psi().adjoint() * z, weights());
}

}} /* sopt::proximal */

#endif
