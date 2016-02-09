#ifndef SOPT_PROXIMAL_EXPRESSION_H
#define SOPT_PROXIMAL_EXPRESSION_H

#include "sopt/config.h"
#include <iostream>
#include <type_traits>
#include <Eigen/Core>
#include "sopt/utility.h"

namespace sopt {
//! Holds some standard proximals
namespace proximal {

namespace details {
//! \brief Expression referencing a lazy proximal function call
//! \details It helps transform the call ``proximal(out, gamma, input)``
//! to ``out = proximal(gamma, input)`` without incurring copy or allocation overhead if ``out``
//! already exists.
template <class FUNCTION, class DERIVED>
class DelayedProximalFunction
    : public Eigen::ReturnByValue<DelayedProximalFunction<FUNCTION, DERIVED>> {
public:
  typedef typename DERIVED::PlainObject PlainObject;
  typedef typename DERIVED::Index Index;
  typedef typename real_type<typename DERIVED::Scalar>::type Real;

  DelayedProximalFunction(FUNCTION const &func, Real const &gamma, DERIVED const &x)
      : func(func), gamma(gamma), x(x) {}
  DelayedProximalFunction(DelayedProximalFunction const &c)
      : func(c.func), gamma(c.gamma), x(c.x) {}
  DelayedProximalFunction(DelayedProximalFunction &&c)
      : func(std::move(c.func)), gamma(c.gamma), x(c.x) {}

  template <class DESTINATION> void evalTo(DESTINATION &destination) const {
    destination.resizeLike(x);
    func(destination, gamma, x);
  }

  Index rows() const { return x.rows(); }
  Index cols() const { return x.cols(); }

private:
  FUNCTION const func;
  Real const gamma;
  DERIVED const &x;
};

//! \brief Expression referencing a lazy function call to envelope proximal
//! \details It helps transform the call ``proximal(out, input)``
//! to ``out = proximal(input)`` without incurring copy or allocation overhead if ``out``
//! already exists.
template <class FUNCTION, class DERIVED>
class DelayedProximalEnveloppeFunction
    : public Eigen::ReturnByValue<DelayedProximalEnveloppeFunction<FUNCTION, DERIVED>> {
public:
  typedef typename DERIVED::PlainObject PlainObject;
  typedef typename DERIVED::Index Index;
  typedef typename real_type<typename DERIVED::Scalar>::type Real;

  DelayedProximalEnveloppeFunction(FUNCTION const &func, DERIVED const &x) : func(func), x(x) {}
  DelayedProximalEnveloppeFunction(DelayedProximalEnveloppeFunction const &c)
      : func(c.func), x(c.x) {}
  DelayedProximalEnveloppeFunction(DelayedProximalEnveloppeFunction &&c)
      : func(std::move(c.func)), x(c.x) {}

  template <class DESTINATION> void evalTo(DESTINATION &destination) const {
    destination.resizeLike(x);
    func(destination, x);
  }

  Index rows() const { return x.rows(); }
  Index cols() const { return x.cols(); }

private:
  FUNCTION const func;
  DERIVED const &x;
};

} /* details */

//! Eigen expression from proximal functions
template <class FUNC, class T0>
using ProximalExpression = details::DelayedProximalFunction<FUNC, Eigen::MatrixBase<T0>>;
//! Eigen expression from proximal enveloppe functions
template <class FUNC, class T0>
using EnveloppeExpression = details::DelayedProximalEnveloppeFunction<FUNC, Eigen::MatrixBase<T0>>;
}
} /* sopt::proximal */

namespace Eigen {
namespace internal {
template <class FUNCTION, class VECTOR>
struct traits<sopt::proximal::details::DelayedProximalFunction<FUNCTION, VECTOR>> {
  typedef typename VECTOR::PlainObject ReturnType;
};
template <class FUNCTION, class VECTOR>
struct traits<sopt::proximal::details::DelayedProximalEnveloppeFunction<FUNCTION, VECTOR>> {
  typedef typename VECTOR::PlainObject ReturnType;
};
}
}

#endif
