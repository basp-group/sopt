#ifndef SOPT_POWER_METHO_H
#define SOPT_POWER_METHO_H

#include "sopt/config.h"
#include <functional>
#include <limits>
#include "sopt/exception.h"
#include "sopt/linear_transform.h"
#include "sopt/logging.h"
#include "sopt/types.h"

namespace sopt {
namespace algorithm {

//! \brief Computes an upper bound on the norm of an operator
template <class SCALAR> class PowerMethod {
public:
  //! Scalar type
  typedef SCALAR value_type;
  //! Scalar type
  typedef value_type Scalar;
  //! Real type
  typedef typename real_type<Scalar>::type Real;
  //! Type of then underlying vectors
  typedef Vector<Scalar> t_Vector;
  //! Type of the Ψ and Ψ^H operations, as well as Φ and Φ^H
  typedef LinearTransform<t_Vector> t_LinearTransform;

  //! Holds result vector as well
  struct DiagnosticAndResult {
    //! Number of iterations
    t_uint niters;
    //! Wether convergence was achieved
    bool good;
    //! Output x
    Scalar bound;
  };

  //! Setups ProximalADMM
  PowerMethod() : itermax_(std::numeric_limits<t_uint>::max()), tolerance_(1e-8) {}
  virtual ~PowerMethod() {}

// Macro helps define properties that can be initialized as in
// auto sdmm  = ProximalADMM<float>().prop0(value).prop1(value);
#define SOPT_MACRO(NAME, TYPE)                                                                     \
  TYPE const &NAME() const { return NAME##_; }                                                     \
  PowerMethod<SCALAR> &NAME(TYPE const &NAME) {                                                    \
    NAME##_ = NAME;                                                                                \
    return *this;                                                                                  \
  }                                                                                                \
                                                                                                   \
protected:                                                                                         \
  TYPE NAME##_;                                                                                    \
                                                                                                   \
public:

  //! Maximum number of iterations
  SOPT_MACRO(itermax, t_uint);
  //! Convergence criteria
  SOPT_MACRO(tolerance, Real);
#undef SOPT_MACRO
  //! \brief Calls the power method
  DiagnosticAndResult operator()(t_LinearTransform const &A, t_Vector const &input) const;
  //! \brief Calls the power method
  //! \details Takes an integer giving the size of the input vector. The linear transform may or may
  //! not have a fixed size per se. E.g. an FFT is the same for any size vector.
  DiagnosticAndResult operator()(t_LinearTransform const &A, t_uint const &size) const {
    return operator()(A, t_Vector::Random(size));
  }
  //! \brief Calls the power method
  //! \detail Only works for fixed-size linear transforms
  DiagnosticAndResult operator()(t_LinearTransform const &A) const {
    if(A.sizes()[0] != 0 or A.sizes()[1] != 0)
      SOPT_THROW("The input transform does not have a fixed size"
                 ": please give size or vector on input");
    return operator()(A, t_Vector::Random(A.size()[2]));
  }

  //! \brief Computes an upper bound on an operator
  //! \details This version will throw if convergence is not achieved
  Scalar compute(t_LinearTransform const &A, t_Vector const &input) const {
    auto const result = operator()(A, input);
    if(not result.good)
      SOPT_THROW("Did not converge in " << itermax() << " iterations");
    return result.bound;
  }
  //! \brief Calls the power method
  //! \details Takes an integer giving the size of the input vector. The linear transform may or may
  //! not have a fixed size per se. E.g. an FFT is the same for any size vector.
  DiagnosticAndResult compute(t_LinearTransform const &A, t_uint const &size) const {
    return compute(A, t_Vector::Random(size));
  }
  //! \brief Calls the power method
  //! \detail Only works for fixed-size linear transforms
  DiagnosticAndResult compute(t_LinearTransform const &A) const {
    if(A.sizes()[0] != 0 or A.sizes()[1] != 0)
      SOPT_THROW("The input transform does not have a fixed size"
                 ": please give size or vector on input");
    return compute(A, t_Vector::Random(A.size()[2]));
  }

protected:
};

template <class SCALAR>
typename PowerMethod<SCALAR>::DiagnosticAndResult PowerMethod<SCALAR>::
operator()(t_LinearTransform const &A, t_Vector const &input) const {
  SOPT_INFO("Computing the upper bound of a given operator");

  t_Vector x = input.normalized();
  auto previous_bound = 1;

  for(t_uint niters(0); niters < itermax(); ++niters) {
    x = A.adjoint() * (A * x);
    auto const bound = x.stableNorm() / static_cast<Real>(x.size());
    auto const rel_val = std::abs((bound - previous_bound) / previous_bound);
    SOPT_NOTICE("    - Iteration {}/{} -- norm: {}", niters, itermax(), bound);

    if(rel_val < tolerance()) {
      SOPT_INFO("    - converged in {} of {} iterations", niters, itermax());
      return {niters, true, bound};
    }

    x /= bound;
    previous_bound = bound;
  }
  // check function exists, otherwise, don't know if convergence is meaningful
  SOPT_WARN("    - did not converge within {} iterations", itermax());
  return {itermax(), false, previous_bound};
}
}
} /* sopt::algorithm */
#endif
