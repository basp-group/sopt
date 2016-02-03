#ifndef SOPT_L1_PROXIMAL_H
#define SOPT_L1_PROXIMAL_H

#include <Eigen/Core>
#include <array>
#include <iostream>
#include <type_traits>

#include "sopt/linear_transform.h"
#include "sopt/proximal_expression.h"
#include "sopt/utility.h"

namespace sopt {
namespace proximal {

//! \brief L1 proximal, including linear transform
//! \details This function computes the prox operator of the l1
//!  norm for the input vector \f$x\f$. It solves the problem:
//!  \f[ min_{z} 0.5||x - z||_2^2 + γ ||Ψ^† z||_w1 \f]
//!  where \f$Ψ \in C^{N_x \times N_r} \f$ is the sparsifying operator, and \f[|| ||_w1\f] is the
//!  weighted L1 norm.
template <class SCALAR> class L1TightFrame {
public:
  //! Underlying scalar type
  typedef SCALAR Scalar;
  //! Underlying real scalar type
  typedef typename real_type<Scalar>::type Real;

  L1TightFrame()
      : Psi_(linear_transform_identity<Scalar>()), nu_(1e0), weights_(Vector<Real>::Ones(1)) {}

#define SOPT_MACRO(NAME, TYPE)                                                                     \
  TYPE const &NAME() const { return NAME##_; }                                                     \
  L1TightFrame<Scalar> &NAME(TYPE const &NAME) {                                                   \
    NAME##_ = NAME;                                                                                \
    return *this;                                                                                  \
  }                                                                                                \
                                                                                                   \
protected:                                                                                         \
  TYPE NAME##_;                                                                                    \
                                                                                                   \
public:
  //! Linear transform applied to input prior to L1 norm
  SOPT_MACRO(Psi, LinearTransform<Vector<Scalar>>);
  //! Bound on the squared norm of the operator Ψ
  SOPT_MACRO(nu, Real);
#undef SOPT_MACRO
  //! Weights of the l1 norm
  Vector<Real> const &weights() const { return weights_; }
  //! Weights of the l1 norm
  template <class T> L1TightFrame<Scalar> &weights(Eigen::MatrixBase<T> const &w) {
    if((w.array() < 0e0).any())
      SOPT_THROW("Weights cannot be negative");
    if(w.stableNorm() < 1e-12)
      SOPT_THROW("Weights cannot be null");
    weights_ = w;
    return *this;
  }

  //! Set weights to a single value
  L1TightFrame<Scalar> &weights(Real const &value) {
    if(value <= 0e0)
      SOPT_THROW("Weight cannot be negative or null");
    weights_ = Vector<Real>::Ones(1) * value;
    return *this;
  }

  //! Set Ψ and Ψ^† using a matrix
  template <class... ARGS>
  typename std::enable_if<sizeof...(ARGS) >= 1, L1TightFrame &>::type Psi(ARGS &&... args) {
    Psi_ = linear_transform(std::forward<ARGS>(args)...);
    return *this;
  }

  //! Computes proximal for given γ
  template <class T0, class T1>
  typename std::enable_if<is_complex<Scalar>::value == is_complex<typename T0::Scalar>::value
                          and is_complex<Scalar>::value
                                  == is_complex<typename T1::Scalar>::value>::type
  operator()(Eigen::MatrixBase<T0> &out, Real gamma, Eigen::MatrixBase<T1> const &x) const;

  //! Lazy version
  template <class T0>
  ProximalExpression<L1TightFrame<Scalar> const &, T0>
  operator()(Real const &gamma, Eigen::MatrixBase<T0> const &x) const {
    return {*this, gamma, x};
  }

  //! \f[ 0.5||x - z||_2^2 + γ||Ψ^† z||_w1 \f]
  template <class T0, class T1>
  typename std::enable_if<is_complex<Scalar>::value == is_complex<typename T0::Scalar>::value
                              and is_complex<Scalar>::value
                                      == is_complex<typename T1::Scalar>::value,
                          Real>::type
  objective(Eigen::MatrixBase<T0> const &x, Eigen::MatrixBase<T1> const &z,
            Real const &gamma) const;

protected:
  //! Weights associated with the l1-norm
  Vector<Real> weights_;
};

template <class SCALAR>
template <class T0, class T1>
typename std::enable_if<is_complex<SCALAR>::value == is_complex<typename T0::Scalar>::value
                        and is_complex<SCALAR>::value
                                == is_complex<typename T1::Scalar>::value>::type
L1TightFrame<SCALAR>::
operator()(Eigen::MatrixBase<T0> &out, Real gamma, Eigen::MatrixBase<T1> const &x) const {

  Vector<Scalar> const psit_x = Psi().adjoint() * x;
  if(weights().size() == 1)
    out = Psi() * (soft_threshhold(psit_x, nu() * gamma * weights()(0)) - psit_x) / nu() + x;
  else
    out = Psi() * (soft_threshhold(psit_x, nu() * gamma * weights()) - psit_x) / nu() + x;
  SOPT_INFO("Prox L1: objective = {}", objective(x, out, gamma));
}

template <class SCALAR>
template <class T0, class T1>
typename std::enable_if<is_complex<SCALAR>::value == is_complex<typename T0::Scalar>::value
                            and is_complex<SCALAR>::value == is_complex<typename T1::Scalar>::value,
                        typename real_type<SCALAR>::type>::type
L1TightFrame<SCALAR>::objective(Eigen::MatrixBase<T0> const &x, Eigen::MatrixBase<T1> const &z,
                                Real const &gamma) const {
  return 0.5 * (x - z).squaredNorm() + gamma * sopt::l1_norm(Psi().adjoint() * z, weights());
}

//! \brief L1 proximal, including linear transform
//! \details This function computes the prox operator of the l1
//!  norm for the input vector \f$x\f$. It solves the problem:
//!  \f[ min_{z} 0.5||x - z||_2^2 + γ ||Ψ^† z||_w1 \f]
//!  where \f$Ψ \in C^{N_x \times N_r} \f$ is the sparsifying operator, and \f[|| ||_w1\f] is the
//!  weighted L1 norm.
template <class SCALAR> class L1 : protected L1TightFrame<SCALAR> {
public:
  //! Functor to do fista mixing
  class FistaMixing;
  //! Functor to do no mixing
  class NoMixing;
  //! Functor to check convergence and cycling
  class Breaker;

  using L1TightFrame<SCALAR>::objective;

  //! Underlying scalar type
  typedef typename L1TightFrame<SCALAR>::Scalar Scalar;
  //! Underlying real scalar type
  typedef typename L1TightFrame<SCALAR>::Real Real;

  //! How did calling L1 go?
  struct Diagnostic {
    //! Number of iterations
    t_uint niters;
    //! Relative variation of the objective function
    Real relative_variation;
    //! Value of the objective function
    Real objective;
    //! Wether convergence was achieved
    bool good;
    Diagnostic(t_uint niters = 0, Real relative_variation = 0, Real objective = 0,
               bool good = false)
        : niters(niters), relative_variation(relative_variation), objective(objective), good(good) {
    }
  };

  //! Result from calling L1
  struct DiagnosticAndResult : public Diagnostic {
    //! The proximal value
    Vector<SCALAR> proximal;
  };

  //! Computes proximal for given γ
  template <class T0, class T1>
  Diagnostic
  operator()(Eigen::MatrixBase<T0> &out, Real gamma, Eigen::MatrixBase<T1> const &x) const {
    // Note that we *must* call eval on x, in case it is an expression involving out
    if(fista_mixing())
      return operator()(out, gamma, x.eval(), FistaMixing());
    else
      return operator()(out, gamma, x.eval(), NoMixing());
  }

  //! Lazy version
  template <class T0>
  DiagnosticAndResult operator()(Real const &gamma, Eigen::MatrixBase<T0> const &x) const {
    DiagnosticAndResult result;
    static_cast<Diagnostic &>(result) = operator()(result.proximal, gamma, x);
    return result;
  }

  L1()
      : L1TightFrame<SCALAR>(), itermax_(0), tolerance_(1e-8), positivity_constraint_(false),
        real_constraint_(false), fista_mixing_(true) {}

#define SOPT_MACRO(NAME, TYPE)                                                                     \
  TYPE const &NAME() const { return NAME##_; }                                                     \
  L1<Scalar> &NAME(TYPE const &NAME) {                                                             \
    NAME##_ = NAME;                                                                                \
    return *this;                                                                                  \
  }                                                                                                \
                                                                                                   \
protected:                                                                                         \
  TYPE NAME##_;                                                                                    \
                                                                                                   \
public:
  //! \brief Maximum number of iterations before bailing out
  //! \details 0 means algorithm breaks only if convergence is reached.
  SOPT_MACRO(itermax, t_uint);
  //! Tolerance criteria
  SOPT_MACRO(tolerance, Real);
  //! Whether to apply positivity constraints
  SOPT_MACRO(positivity_constraint, bool);
  //! Whether the output should be constrained to be real
  SOPT_MACRO(real_constraint, bool);
  //! Whether to do fista mixing or not
  SOPT_MACRO(fista_mixing, bool);
#undef SOPT_MACRO

  //! Weights of the l1 norm
  Vector<Real> const & weights() const { return L1TightFrame<Scalar>::weights(); }
  //! Set weights to an array of values
  template <class T> L1<Scalar> &weights(Eigen::MatrixBase<T> const &w) {
    L1TightFrame<Scalar>::weights(w);
    return *this;
  }
  //! Set weights to a single value
  L1<Scalar> &weights(Real const &w) {
    L1TightFrame<Scalar>::weights(w);
    return this;
  }

  //! Bounds on the squared norm of the operator Ψ
  Real nu() const { return L1TightFrame<Scalar>::nu(); }
  //! Sets the bound on the squared norm of the operator Ψ
  L1<Scalar> &nu(Real const &nu) {
    L1TightFrame<SCALAR>::nu(nu);
    return *this;
  }

  //! Linear transform applied to input prior to L1 norm
  LinearTransform<Vector<Scalar>> const & Psi() const { return L1TightFrame<Scalar>::Psi(); }
  //! Set Ψ and Ψ^† using a matrix
  template <class... ARGS>
  typename std::enable_if<sizeof...(ARGS) >= 1, L1<Scalar> &>::type Psi(ARGS &&... args) {
    L1TightFrame<Scalar>::Psi(std::forward<ARGS>(args)...);
    return *this;
  }

  //! \brief Special case if Ψ ia a tight frame.
  //! \see L1TightFrame
  template <class... T>
  auto tight_frame(T &&... args) const
      -> decltype(this->L1TightFrame<Scalar>::operator()(std::forward<T>(args)...)) {
    return this->L1TightFrame<Scalar>::operator()(std::forward<T>(args)...);
  }

protected:
  //! Applies one or another soft-threshhold, depending on weight
  template <class T1>
  Vector<SCALAR> apply_soft_threshhold(Real gamma, Eigen::MatrixBase<T1> const &x) const;
  //! Applies constraints to input expression
  template <class T0, class T1>
  void apply_constraints(Eigen::MatrixBase<T0> &out, Eigen::MatrixBase<T1> const &x) const;

  //! Operation with explicit mixing step
  template <class T0, class T1, class MIXING>
  Diagnostic operator()(Eigen::MatrixBase<T0> &out, Real gamma, Eigen::MatrixBase<T1> const &x,
                        MIXING mixing) const;
};

//! Computes proximal for given γ
template <class SCALAR>
template <class T0, class T1, class MIXING>
typename L1<SCALAR>::Diagnostic L1<SCALAR>::
operator()(Eigen::MatrixBase<T0> &out, Real gamma, Eigen::MatrixBase<T1> const &x,
           MIXING mixing) const {

  SOPT_NOTICE("  Starting Proximal L1 operator:");
  t_uint niters = 0;
  out = x;

  Breaker breaker(objective(x, x, gamma), tolerance(), not fista_mixing());
  SOPT_NOTICE("    - iter {}, prox_fval = {}", niters, breaker.current());
  Vector<Scalar> const res = Psi().adjoint() * out;
  Vector<Scalar> u_l1 = 1e0 / nu() * (res - apply_soft_threshhold(gamma, res));
  apply_constraints(out, x - Psi() * u_l1);

  // Move on to other iterations
  for(++niters; niters < itermax() or itermax() == 0; ++niters) {

    auto const do_break = breaker(objective(x, out, gamma));
    SOPT_NOTICE("    - iter {}, prox_fval = {}, rel_fval = {}", niters, breaker.current(),
                breaker.relative_variation());
    if(do_break)
      break;

    Vector<Scalar> const res = u_l1 * nu() + Psi().adjoint() * out;
    mixing(u_l1, 1e0 / nu() * (res - apply_soft_threshhold(gamma, res)), niters);
    apply_constraints(out, x - Psi() * u_l1);
  }

  if(breaker.two_cycle())
    SOPT_NOTICE("Two-cycle detected when computing L1");

  if(breaker.converged()) {
    SOPT_INFO("  Proximal L1 operator converged at {} in {} iterations", breaker.current(), niters);
  } else
    SOPT_INFO("  Proximal L1 operator did not converge after {} iterations", niters);
  return {niters, breaker.relative_variation(), breaker.current(), breaker.converged()};
}

template <class SCALAR>
template <class T1>
Vector<SCALAR> L1<SCALAR>::apply_soft_threshhold(Real gamma, Eigen::MatrixBase<T1> const &x) const {
  if(weights().size() == 1)
    return soft_threshhold(x, gamma * weights()(0));
  else
    return soft_threshhold(x, gamma * weights());
}

template <class SCALAR>
template <class T0, class T1>
void L1<SCALAR>::apply_constraints(Eigen::MatrixBase<T0> &out,
                                   Eigen::MatrixBase<T1> const &x) const {
  if(positivity_constraint())
    out = sopt::positive_quadrant(x);
  else if(real_constraint())
    out = x.real().template cast<SCALAR>();
  else
    out = x;
}

template <class SCALAR> class L1<SCALAR>::FistaMixing {
public:
  typedef typename real_type<SCALAR>::type Real;
  FistaMixing() : t(1){};
  template <class T1>
  void operator()(Vector<SCALAR> &previous, Eigen::MatrixBase<T1> const &unmixed, t_uint iter) {
    // reset
    if(iter == 0) {
      previous = unmixed;
      return;
    }
    if(iter <= 1)
      t = next(1);
    auto const prior_t = t;
    t = next(t);
    auto const alpha = (prior_t - 1) / t;
    previous = (1e0 + alpha) * unmixed.derived() - alpha * previous;
  }
  static Real next(Real t) { return 0.5 + 0.5 * std::sqrt(1e0 + 4e0 * t * t); }

private:
  Real t;
};

template <class SCALAR> class L1<SCALAR>::NoMixing {
public:
  template <class T1>
  void operator()(Vector<SCALAR> &previous, Eigen::MatrixBase<T1> const &unmixed, t_uint) {
    previous = unmixed;
  }
};

template <class SCALAR> class L1<SCALAR>::Breaker {
public:
  typedef typename real_type<SCALAR>::type Real;
  //! Constructs a breaker object
  //! \param[in] objective: the first objective function
  //! \param[in] tolerance: Convergence criteria for convergence
  //! \param[in] do_two_cycle: Whether to enable two cycle detections. Only necessary when mixing
  //! is not enabled.
  Breaker(Real objective, Real tolerance = 1e-8, bool do_two_cycle = true)
      : tolerance_(tolerance), iter(0), objectives({{objective, 0, 0, 0}}),
        do_two_cycle(do_two_cycle) {}
  //! True if we should break out of loop
  bool operator()(Real objective) {
    ++iter;
    objectives = {{objective, objectives[0], objectives[1], objectives[2]}};
    return converged() or two_cycle();
  }
  //! Current objective
  Real current() const { return objectives[0]; }
  //! Current objective
  Real previous() const { return objectives[1]; }
  //! Variation in the objective function
  Real relative_variation() const { return std::abs((current() - previous()) / current()); }
  //! \brief Whether we have a cycle of period two
  //! \details Cycling is prone to happen without mixing, it seems.
  bool two_cycle() const {
    return do_two_cycle and iter > 3 and std::abs(objectives[0] - objectives[2]) < tolerance()
           and std::abs(objectives[1] - objectives[3]) < tolerance();
  }

  //! True if relative variation smaller than tolerance
  bool converged() const {
    // If current ~ 0, then defaults to absolute convergence
    // This is mainly to avoid a division by zero
    if(std::abs(current() * 1000) < tolerance())
      return std::abs(previous() * 1000) < tolerance();
    return relative_variation() < tolerance();
  }
  //! Tolerance criteria
  Real tolerance() const { return tolerance_; }
  //! Tolerance criteria
  L1<SCALAR>::Breaker &tolerance(Real tol) const {
    tolerance_ = tol;
    return *this;
  }

protected:
  Real tolerance_;
  t_uint iter;
  std::array<Real, 4> objectives;
  bool do_two_cycle;
};
}
} /* sopt::proximal */

#endif
