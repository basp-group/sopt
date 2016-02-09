#ifndef SOPT_L1_PROXIMAL_ADMM_H
#define SOPT_L1_PROXIMAL_ADMM_H

#include "sopt/config.h"
#include <numeric>
#include <utility>
#include "sopt/exception.h"
#include "sopt/l1_proximal.h"
#include "sopt/linear_transform.h"
#include "sopt/logging.h"
#include "sopt/padmm.h"
#include "sopt/proximal.h"
#include "sopt/types.h"

namespace sopt {
namespace algorithm {

//! \brief Specialization of Proximal ADMM for Purify
//! \details \f$\min_{x, z} f(x) + h(z)\f$ subject to \f$Φx + z = y\f$, where \f$f(x) =
//! ||Ψ^Hx||_1 + i_C(x)\f$ and \f$h(x) = i_B(z)\f$ with \f$C = R^N_{+}\f$ and \f$B = {z \in R^M:
//! ||z||_2 \leq \epsilon}\f$
template <class SCALAR> class L1ProximalADMM : private ProximalADMM<SCALAR> {
  //! Defines convergence behaviour
  struct Breaker;

public:
  //! Scalar type
  typedef typename ProximalADMM<SCALAR>::value_type value_type;
  typedef typename ProximalADMM<SCALAR>::Scalar Scalar;
  typedef typename ProximalADMM<SCALAR>::Real Real;
  typedef typename ProximalADMM<SCALAR>::t_Vector t_Vector;
  typedef typename ProximalADMM<SCALAR>::t_LinearTransform t_LinearTransform;
  typedef typename ProximalADMM<SCALAR>::t_Proximal t_Proximal;

  //! Values indicating how the algorithm ran
  struct Diagnostic : public ProximalADMM<Scalar>::Diagnostic {
    //! Diagnostic from calling L1 proximal
    typename proximal::L1<Scalar>::Diagnostic l1_diag;
    Diagnostic(t_uint niters = 0u, bool good = false,
               typename proximal::L1<Scalar>::Diagnostic const &l1diag
               = typename proximal::L1<Scalar>::Diagnostic())
        : ProximalADMM<Scalar>::Diagnostic(niters, good), l1_diag(l1diag) {}
  };
  //! Holds result vector as well
  struct DiagnosticAndResult : public Diagnostic {
    //! Output x
    t_Vector x;
  };

  L1ProximalADMM()
      : ProximalADMM<SCALAR>(nullptr, nullptr), l1_proximal_(), l2ball_proximal_(1e0),
        tight_frame_(false), relative_variation_(1e-4), residual_convergence_(1e-4) {
    set_f_and_g_proximal_to_members_of_this();
  }
  L1ProximalADMM(L1ProximalADMM<Scalar> const &c)
      : ProximalADMM<Scalar>(c), l1_proximal_(c.l1_proximal_), l2ball_proximal_(c.l2ball_proximal_),
        tight_frame_(c.tight_frame_), relative_variation_(c.relative_variation_),
        residual_convergence_(c.residual_convergence_) {
    set_f_and_g_proximal_to_members_of_this();
  }
  L1ProximalADMM(L1ProximalADMM<Scalar> &&c)
      : ProximalADMM<Scalar>(std::move(c)), l1_proximal_(std::move(c.l1_proximal_)),
        l2ball_proximal_(std::move(c.l2ball_proximal_)), tight_frame_(c.tight_frame_),
        relative_variation_(c.relative_variation_), residual_convergence_(c.residual_convergence_) {
    set_f_and_g_proximal_to_members_of_this();
  }

  void operator=(L1ProximalADMM<Scalar> const &c) {
    ProximalADMM<Scalar>::operator=(c);
    l1_proximal_ = c.l1_proximal_;
    l2ball_proximal_ = c.l2ball_proximal_;
    tight_frame_ = c.tight_frame_;
    set_f_and_g_proximal_to_members_of_this();
  }
  void operator=(L1ProximalADMM<Scalar> &&c) {
    ProximalADMM<Scalar>::operator=(std::move(c));
    l1_proximal_ = std::move(c.l1_proximal_);
    l2ball_proximal_ = std::move(c.l2ball_proximal_);
    tight_frame_ = std::move(c.tight_frame_);
    set_f_and_g_proximal_to_members_of_this();
  }

  virtual ~L1ProximalADMM() {}

// Macro helps define properties that can be initialized as in
// auto sdmm  = ProximalADMM<float>().prop0(value).prop1(value);
#define SOPT_MACRO(NAME, TYPE, CODE)                                                               \
  TYPE const &NAME() const { return NAME##_; }                                                     \
  L1ProximalADMM<SCALAR> &NAME(TYPE const &NAME) {                                                 \
    NAME##_ = NAME;                                                                                \
    CODE;                                                                                          \
    return *this;                                                                                  \
  }                                                                                                \
                                                                                                   \
protected:                                                                                         \
  TYPE NAME##_;                                                                                    \
                                                                                                   \
public:

  //! The L1 proximal functioning as f
  SOPT_MACRO(l1_proximal, proximal::L1<Scalar>, set_f_and_g_proximal_to_members_of_this());
  //! The weighted L2 proximal functioning as g
  SOPT_MACRO(l2ball_proximal, proximal::WeightedL2Ball<Scalar>,
             set_f_and_g_proximal_to_members_of_this());
  //! Whether Ψ is a tight-frame or not
  SOPT_MACRO(tight_frame, bool, );
  //! \brief Convergence of the relative variation of the objective functions
  //! \details If negative, this convergence criteria is disabled.
  SOPT_MACRO(relative_variation, Real, );
  //! \brief Convergence of the residuals
  //! \details If negative, this convergence criteria is disabled.
  SOPT_MACRO(residual_convergence, Real, );
#undef SOPT_MACRO

  //! \brief Analysis operator Ψ
  //! \details Under-the-hood, the object is actually owned by the L1 proximal.
  t_LinearTransform const &Psi() const { return l1_proximal().Psi(); }
  //! Analysis operator Ψ
  template <class... ARGS>
  typename std::enable_if<sizeof...(ARGS) >= 1, L1ProximalADMM<Scalar> &>::type
  Psi(ARGS &&... args) {
    l1_proximal().Psi(std::forward<ARGS>(args)...);
    return *this;
  }
  //! Φ initialized via some call to \ref linear_transform
  template <class... ARGS>
  typename std::enable_if<sizeof...(ARGS) >= 1, L1ProximalADMM<Scalar> &>::type
  Phi(ARGS &&... args) {
    ProximalADMM<Scalar>::Phi(std::forward<ARGS>(args)...);
    return *this;
  }

  //! \brief L1 proximal used during calculation
  //! \details Non-const version to setup the object.
  proximal::L1<Scalar> &l1_proximal() { return l1_proximal_; }
  //! \brief Proximal of the L2 ball
  //! \details Non-const version to setup the object.
  proximal::WeightedL2Ball<Scalar> &l2ball_proximal() { return l2ball_proximal_; }

  //! Type-erased version of the l1 proximal
  t_Proximal const &f_proximal() { return ProximalADMM<Scalar>::f_proximal(); }
  //! Type-erased version of the l2 proximal
  t_Proximal const &g_proximal() { return ProximalADMM<Scalar>::g_proximal(); }

// Forwards get/setters to L1 and L2Ball proximals
// In practice, we end up with a bunch of functions that make it simpler to set or get values
// associated with the two proximal operators.
// E.g.: `paddm.l1_proximal_itermax(100).l2ball_epsilon(1e-2).l1_proximal_tolerance(1e-4)`.
// ~~~
#define SOPT_MACRO(VAR, NAME, PROXIMAL)                                                            \
  /** \brief Forwards to l1_proximal **/                                                           \
  decltype(std::declval<proximal::PROXIMAL<Scalar> const>().VAR()) NAME##_proximal_##VAR() const { \
    return NAME##_proximal().VAR();                                                                \
  }                                                                                                \
  /** \brief Forwards to l1_proximal **/                                                           \
  L1ProximalADMM<Scalar> &NAME##_proximal_##VAR(                                                   \
      decltype(std::declval<proximal::PROXIMAL<Scalar> const>().VAR()) VAR) {                      \
    NAME##_proximal().VAR(VAR);                                                                    \
    return *this;                                                                                  \
  }
  SOPT_MACRO(itermax, l1, L1);
  SOPT_MACRO(tolerance, l1, L1);
  SOPT_MACRO(positivity_constraint, l1, L1);
  SOPT_MACRO(real_constraint, l1, L1);
  SOPT_MACRO(fista_mixing, l1, L1);
  SOPT_MACRO(nu, l1, L1);
  SOPT_MACRO(weights, l1, L1);
  SOPT_MACRO(epsilon, l2ball, WeightedL2Ball);
  SOPT_MACRO(weights, l2ball, WeightedL2Ball);
#undef SOPT_MACRO

// Includes getters and redefines setters to return this object
#define SOPT_MACRO(NAME)                                                                           \
  using ProximalADMM<Scalar>::NAME;                                                                \
  /** \brief Forwards to ProximalADMM base class **/                                               \
  L1ProximalADMM<Scalar> &NAME(decltype(std::declval<ProximalADMM<Scalar>>().NAME()) NAME) {       \
    ProximalADMM<Scalar>::NAME(NAME);                                                              \
    return *this;                                                                                  \
  }
  SOPT_MACRO(itermax);
  SOPT_MACRO(gamma);
  SOPT_MACRO(nu);
  SOPT_MACRO(lagrange_update_scale);
  SOPT_MACRO(Phi);
  SOPT_MACRO(is_converged);
#undef SOPT_MACRO

  //! Calls l1 proximal operator, checking for real constraints and tight frame
  template <class T0, class T1>
  typename proximal::L1<Scalar>::Diagnostic
  l1_proximal(Eigen::MatrixBase<T0> &out, Real gamma, Eigen::MatrixBase<T1> const &x) const {
    return l1_proximal_real_constraint() ? call_l1_proximal(out, gamma, x.real()) :
                                           call_l1_proximal(out, gamma, x);
  }

  //! Forwards call to weighted L2 ball proximal
  template <class T>
  auto l2ball_proximal(Eigen::MatrixBase<T> const &x) const
      -> decltype(std::declval<proximal::WeightedL2Ball<Scalar> const>()(Real(0), x)) {
    return l2ball_proximal()(Real(0), x);
  }

  //! \brief Call Proximal ADMM for L1 and L2 ball
  //! \param[out] out: Output x vector
  //! \param[in] input: Target measurements
  Diagnostic operator()(t_Vector &out, t_Vector const &input) const;
  //! \brief Calls Proximal ADMM for L1 and L2 ball
  //! \param[in] input: Target measurement vector y
  DiagnosticAndResult operator()(t_Vector const &input) const {
    DiagnosticAndResult result;
    static_cast<Diagnostic &>(result) = operator()(result.x, input);
    return result;
  }

protected:
  //! Keeps track of the last call to the L1 proximal
  mutable typename proximal::L1<Scalar>::Diagnostic l1_diagnostic;
  //! Calls l1 proximal operator, checking for thight frame
  template <class T0, class T1>
  typename proximal::L1<Scalar>::Diagnostic
  call_l1_proximal(Eigen::MatrixBase<T0> &out, Real gamma, Eigen::MatrixBase<T1> const &x) const {
    if(tight_frame()) {
      l1_proximal().tight_frame(out, gamma, x);
      return {0, 0, l1_proximal().objective(x, out, gamma), true};
    }
    return l1_proximal()(out, gamma, x);
  }

  //! Sets the result from this call to L1 proximal so it can be used later
  void erased_f_proximal(t_Vector &out, Real gamma, t_Vector const &x) const {
    l1_diagnostic = l1_proximal(out, gamma, x);
  }

  //! References
  void set_f_and_g_proximal_to_members_of_this() {
    using namespace std::placeholders;
    ProximalADMM<Scalar>::f_proximal(
        std::bind(&L1ProximalADMM<Scalar>::erased_f_proximal, this, _1, _2, _3));
    ProximalADMM<Scalar>::g_proximal(std::cref(l2ball_proximal()));
  }
};

template <class SCALAR>
typename L1ProximalADMM<SCALAR>::Diagnostic L1ProximalADMM<SCALAR>::
operator()(t_Vector &out, t_Vector const &input) const {

  SOPT_INFO("Performing Proximal ADMM with L1 and L2 operators");
  ProximalADMM<Scalar>::sanity_check(input);

  t_Vector lambda = t_Vector::Zero(input.size()), z = t_Vector::Zero(input.size()),
           residual = t_Vector::Zero(input.size());
  bool const has_user_convergence = static_cast<bool>(is_converged());
  l1_diagnostic = {0, 0, 0, false};

  SOPT_NOTICE("    - Initialization");
  ProximalADMM<Scalar>::initialization_step(input, out, residual);
  std::pair<Real, Real> objectives{sopt::l1_norm(residual, l1_proximal_weights()), 0};

  for(t_uint niters(0); niters < itermax(); ++niters) {
    SOPT_NOTICE("    - Iteration {}/{}. ", niters, itermax());
    ProximalADMM<Scalar>::iteration_step(input, out, residual, lambda, z);

    // print-out stuff
    objectives.second = objectives.first;
    objectives.first = sopt::l1_norm(residual, l1_proximal_weights());
    t_real const relative_objective
        = std::abs(objectives.first - objectives.second) / objectives.first;
    auto const residual_norm = sopt::l2_norm(residual, l2ball_proximal_weights());

    SOPT_NOTICE("    - objective: obj value = {}, rel obj = {}", objectives.first,
                relative_objective);
    SOPT_NOTICE("    - Residuals: epsilon = {}, residual norm = {}", l2ball_proximal_epsilon(),
                residual_norm);

    // convergence stuff
    auto const user = (not has_user_convergence) or is_converged(out);
    auto const res = residual_convergence() <= 0e0 or residual_norm < residual_convergence();
    auto const rel = relative_variation() <= 0e0 or relative_objective < relative_variation();
    if(user and res and rel) {
      SOPT_INFO("    - converged in {} of {} iterations", niters, itermax());
      return Diagnostic{niters, true, l1_diagnostic};
    }
  }

  SOPT_WARN("    - did not converge within {} iterations", itermax());
  return {itermax(), false, l1_diagnostic};
}
}
} /* sopt::algorithm */
#endif
