#ifndef SOPT_PROXIMAL_ADMM_H
#define SOPT_PROXIMAL_ADMM_H

#include "sopt/config.h"
#include <functional>
#include <limits>
#include "sopt/exception.h"
#include "sopt/linear_transform.h"
#include "sopt/logging.h"
#include "sopt/types.h"

namespace sopt {
namespace algorithm {

//! \brief Proximal Alternate Direction method of mutltipliers
//! \details \f$\min_{x, z} f(x) + h(z)\f$ subject to \f$Φx + z = y\f$. \f$y\f$ is a target vector.
template <class SCALAR> class ProximalADMM {
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
  //! Type of the convergence function
  typedef ConvergenceFunction<Scalar> t_IsConverged;
  //! Type of the convergence function
  typedef ProximalFunction<Scalar> t_Proximal;

  //! Values indicating how the algorithm ran
  struct Diagnostic {
    //! Number of iterations
    t_uint niters;
    //! Wether convergence was achieved
    bool good;
    //! the residual from the last iteration
    t_Vector residual;

    Diagnostic(t_uint niters = 0u, bool good = false)
        : niters(niters), good(good), residual(t_Vector::Zero(0)) {}
    Diagnostic(t_uint niters, bool good, t_Vector &&residual)
        : niters(niters), good(good), residual(std::move(residual)) {}
  };
  //! Holds result vector as well
  struct DiagnosticAndResult : public Diagnostic {
    //! Output x
    t_Vector x;
  };

  //! Setups ProximalADMM
  //! \param[in] f_proximal: proximal operator of the \f$f\f$ function.
  //! \param[in] g_proximal: proximal operator of the \f$g\f$ function
  template <class DERIVED>
  ProximalADMM(t_Proximal const &f_proximal, t_Proximal const &g_proximal,
               Eigen::MatrixBase<DERIVED> const &target)
      : itermax_(std::numeric_limits<t_uint>::max()), gamma_(1e-8), nu_(1),
        lagrange_update_scale_(0.9), is_converged_(), Phi_(linear_transform_identity<Scalar>()),
        f_proximal_(f_proximal), g_proximal_(g_proximal), target_(target) {}
  virtual ~ProximalADMM() {}

// Macro helps define properties that can be initialized as in
// auto sdmm  = ProximalADMM<float>().prop0(value).prop1(value);
#define SOPT_MACRO(NAME, TYPE)                                                                     \
  TYPE const &NAME() const { return NAME##_; }                                                     \
  ProximalADMM<SCALAR> &NAME(TYPE const &NAME) {                                                   \
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
  //! γ parameter
  SOPT_MACRO(gamma, Real);
  //! ν parameter
  SOPT_MACRO(nu, Real);
  //! Lagrange update scale β
  SOPT_MACRO(lagrange_update_scale, Real);
  //! A function verifying convergence
  SOPT_MACRO(is_converged, t_IsConverged);
  //! Measurement operator
  SOPT_MACRO(Phi, t_LinearTransform);
  //! First proximal
  SOPT_MACRO(f_proximal, t_Proximal);
  //! Second proximal
  SOPT_MACRO(g_proximal, t_Proximal);
#undef SOPT_MACRO
  //! \brief Simplifies calling the proximal of f.
  void f_proximal(t_Vector &out, Real gamma, t_Vector const &x) const {
    f_proximal()(out, gamma, x);
  }
  //! \brief Simplifies calling the proximal of f.
  void g_proximal(t_Vector &out, Real gamma, t_Vector const &x) const {
    g_proximal()(out, gamma, x);
  }

  //! Vector of target measurements
  t_Vector const &target() const { return target_; }
  //! Sets the vector of target measurements
  template <class DERIVED> ProximalADMM<DERIVED> &target(Eigen::MatrixBase<DERIVED> const &target) {
    target_ = target;
    return *this;
  }

  //! Facilitates call to user-provided convergence function
  bool is_converged(t_Vector const &x) const {
    return static_cast<bool>(is_converged()) and is_converged()(x);
  }

  //! \brief Calls Proximal ADMM
  //! \param[out] out: Output vector x
  Diagnostic operator()(t_Vector &out) const { return operator()(out, initial_guess()); }
  //! \brief Calls Proximal ADMM
  //! \param[out] out: Output vector x
  //! \param[in] guess: initial guess
  Diagnostic operator()(t_Vector &out, std::tuple<t_Vector, t_Vector> const &guess) const {
    return operator()(out, std::get<0>(guess), std::get<1>(guess));
  }
  //! \brief Calls Proximal ADMM
  //! \param[in] guess: initial guess
  DiagnosticAndResult operator()(std::tuple<t_Vector, t_Vector> const &guess) const {
    DiagnosticAndResult result;
    static_cast<Diagnostic &>(result) = operator()(result.x, guess);
    return result;
  }
  //! \brief Calls Proximal ADMM
  //! \param[in] guess: initial guess
  DiagnosticAndResult operator()() const {
    DiagnosticAndResult result;
    static_cast<Diagnostic &>(result) = operator()(result.x, initial_guess());
    return result;
  }
  //! Makes it simple to chain different calls to PADMM
  DiagnosticAndResult operator()(DiagnosticAndResult const &warmstart) const {
    DiagnosticAndResult result = warmstart;
    static_cast<Diagnostic &>(result) = operator()(result.x, warmstart.x, warmstart.residual);
    return result;
  }
  //! Set Φ and Φ^† using arguments that sopt::linear_transform understands
  template <class... ARGS>
  typename std::enable_if<sizeof...(ARGS) >= 1, ProximalADMM &>::type Phi(ARGS &&... args) {
    Phi_ = linear_transform(std::forward<ARGS>(args)...);
    return *this;
  }

  //! \brief Computes initial guess for x and the residual using the targets
  //! \details with y the vector of measurements
  //! - x = Φ^T y / ν
  //! - residuals = Φ x - y
  std::tuple<t_Vector, t_Vector> initial_guess() const {
    std::tuple<t_Vector, t_Vector> guess;
    std::get<0>(guess) = Phi().adjoint() * target() / nu();
    std::get<1>(guess) = Phi() * std::get<0>(guess) - target();
    return guess;
  }

protected:
  void iteration_step(t_Vector &out, t_Vector &residual, t_Vector &lambda, t_Vector &z) const;

  //! Checks input makes sense
  void sanity_check(t_Vector const &x_guess, t_Vector const &res_guess) const {
    if((Phi().adjoint() * target()).size() != x_guess.size())
      SOPT_THROW("target, adjoint measurement operator and input vector have inconsistent sizes");
    if(target().size() != res_guess.size())
      SOPT_THROW("target and residual vector have inconsistent sizes");
    if((Phi() * x_guess).size() != target().size())
      SOPT_THROW("target, measurement operator and input vector have inconsistent sizes");
    if(not static_cast<bool>(is_converged()))
      SOPT_WARN("No convergence function was provided: algorithm will run for {} steps", itermax());
  }

  //! \brief Calls Proximal ADMM
  //! \param[out] out: Output vector x
  //! \param[in] guess: initial guess
  //! \param[in] residuals: initial residuals
  Diagnostic operator()(t_Vector &out, t_Vector const &guess, t_Vector const &res) const;

  //! Vector of measurements
  t_Vector target_;
};

template <class SCALAR>
void ProximalADMM<SCALAR>::iteration_step(t_Vector &out, t_Vector &residual, t_Vector &lambda,
                                          t_Vector &z) const {
  g_proximal(z, gamma(), -lambda - residual);
  f_proximal(out, gamma() / nu(), out - Phi().adjoint() * (residual + lambda + z) / nu());
  residual = Phi() * out - target();
  lambda += lagrange_update_scale() * (residual + z);
}

template <class SCALAR>
typename ProximalADMM<SCALAR>::Diagnostic ProximalADMM<SCALAR>::
operator()(t_Vector &out, t_Vector const &x_guess, t_Vector const &res_guess) const {

  SOPT_HIGH_LOG("Performing Proximal ADMM");
  sanity_check(x_guess, res_guess);

  t_Vector lambda = t_Vector::Zero(target().size());
  t_Vector z = t_Vector::Zero(target().size());
  t_Vector residual = res_guess;
  out = x_guess;

  for(t_uint niters(0); niters < itermax(); ++niters) {
    SOPT_LOW_LOG("    - Iteration {}/{}", niters, itermax());
    iteration_step(out, residual, lambda, z);
    SOPT_LOW_LOG("      - Sum of residuals: {}", residual.array().abs().sum());

    if(is_converged(out)) {
      SOPT_MEDIUM_LOG("    - converged in {} of {} iterations", niters, itermax());
      return {niters, true};
    }
  }
  // check function exists, otherwise, don't know if convergence is meaningful
  if(static_cast<bool>(is_converged()))
    SOPT_ERROR("    - did not converge within {} iterations", itermax());
  return {itermax(), false, std::move(residual)};
}
}
} /* sopt::algorithm */
#endif
