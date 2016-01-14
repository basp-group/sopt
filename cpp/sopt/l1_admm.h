#ifndef SOPT_L1_ADMM_H
#define SOPT_L1_ADMM_H

#include <numeric>
#include <utility>

#include "sopt/types.h"
#include "sopt/linear_transform.h"
#include "sopt/proximal.h"
#include "sopt/exception.h"
#include "sopt/logging.h"
#include "sopt/L1_proximal.h"
#include "sopt/admm.h"

namespace sopt { namespace algorithm {


//! \brief Specialization of ADMM for Purify
//! \details \f$\min_{x, z} f(x) + h(z)\f$ subject to \f$Φx + z = y\f$, where \f$f(x) =
//! ||Ψ^Hx||_1 + i_C(x)\f$ and \f$h(x) = i_B(z)\f$ with \f$C = R^N_{+}\f$ and \f$B = {z \in R^M:
//! ||z||_2 \leq \epsilon}\f$.
template<class SCALAR> class L1_ADMM : private ADMM<SCALAR> {
    //! Defines convergence behaviour
    struct Breaker;
  public:
    //! Scalar type
    typedef typename ADMM<SCALAR>::value_type value_type;
    typedef typename ADMM<SCALAR>::Scalar Scalar;
    typedef typename ADMM<SCALAR>::Real Real;
    typedef typename ADMM<SCALAR>::t_Vector t_Vector;
    typedef typename ADMM<SCALAR>::t_LinearTransform t_LinearTransform;
    typedef typename ADMM<SCALAR>::t_Proximal t_Proximal;

    //! Values indicating how the algorithm ran
    struct Diagnostic : public ADMM<Scalar>::Diagnostic {
      //! Diagnostic from calling L1 proximal
      typename proximal::L1<Scalar>::Diagnostic l1_diag;
      Diagnostic(t_uint niters, bool good, typename proximal::L1<Scalar>::Diagnostic const & l1diag)
        : ADMM<Scalar>::Diagnostic(niters, good), l1_diag(l1diag) {}
    };

    L1_ADMM()
      : ADMM<SCALAR>(nullptr, nullptr), l1_proximal_(), l2ball_proximal_(1e0), tight_frame_(false) {
      using namespace std::placeholders;
      ADMM<Scalar>::f_proximal(std::bind(&L1_ADMM<Scalar>::erased_f_proximal, this, _1, _2, _3));
      ADMM<Scalar>::g_proximal(l2ball_proximal());
    }

    virtual ~L1_ADMM() {}

    // Macro helps define properties that can be initialized as in
    // auto sdmm  = ADMM<float>().prop0(value).prop1(value);
#   define SOPT_MACRO(NAME, TYPE)                                                     \
        TYPE const& NAME() const { return NAME ## _; }                                \
        L1_ADMM<SCALAR> & NAME(TYPE const &NAME) { NAME ## _ = NAME; return *this; }  \
      protected:                                                                      \
        TYPE NAME ## _;                                                               \
      public:

    //! The L1 proximal functioning as f
    SOPT_MACRO(l1_proximal, proximal::L1<Scalar>);
    //! The weighted L2 proximal functioning as g
    SOPT_MACRO(l2ball_proximal, proximal::WeightedL2Ball<Scalar>);
    //! Whether Ψ is a tight-frame or not
    SOPT_MACRO(tight_frame, bool);
#   undef SOPT_MACRO

    //! Analysis operator Ψ
    L1_ADMM<Scalar> & Psi(t_LinearTransform const &l) { l1_proximal().Psi(l); return *this; }
    //! \brief Analysis operator Ψ
    //! \details Under-the-hood, the object is actually owned by the L1 proximal.
    t_LinearTransform const & Psi() const { return l1_proximal().Psi(); }
    //! Setup Ψ both here
    //! Ψ initialized via some call to \ref linear_transform
    template<class T0, class ... T>
      typename std::enable_if<
        not std::is_same<typename std::remove_all_extents<T0>::type, t_LinearTransform>::value,
        L1_ADMM<SCALAR> &
      >::type Psi(T0 &&t0, T &&... args) {
        l1_proximal().Psi(linear_transform(std::forward<T0>(t0), std::forward<T>(args)...));
        return *this;
      }
    //! Φ initialized via some call to \ref linear_transform
    template<class T0, class ... T>
      typename std::enable_if<
        not std::is_same<typename std::remove_all_extents<T0>::type, t_LinearTransform>::value,
        L1_ADMM<SCALAR> &
      >::type Phi(T0 &&t0, T &&... args) {
        ADMM<Scalar>::Phi(std::forward<T0>(t0), std::forward<T>(args)...);
        return *this;
      }

    //! \brief L1 proximal used during calculation
    //! \details Non-const version to setup the object.
    proximal::L1<Scalar> & l1_proximal() { return l1_proximal_; }
    //! \brief Proximal of the L2 ball
    //! \details Non-const version to setup the object.
    proximal::WeightedL2Ball<Scalar> & l2ball_proximal() {
      return l2ball_proximal_;
    }

    //! Type-erased version of the l1 proximal
    t_Proximal const & f_proximal() { return ADMM<Scalar>::f_proximal(); }
    //! Type-erased version of the l2 proximal
    t_Proximal const & g_proximal() { return ADMM<Scalar>::g_proximal(); }

    // Forwards get/setters to L1 and L2Ball proximals
    // In practice, we end up with a bunch of functions that make it simpler to set or get values
    // associated with the two proximal operators.
    // E.g.: `paddm.l1_proximal_itermax(100).l2ball_epsilon(1e-2).l1_proximal_tolerance(1e-4)`.
    // ~~~
#   define SOPT_MACRO(VAR, NAME, PROXIMAL)                                                  \
        /** \brief Forwards to l1_proximal **/                                              \
        decltype(std::declval<proximal::PROXIMAL<Scalar> const>().VAR())                    \
        NAME ## _proximal_ ## VAR() const {                                                 \
          return NAME ## _proximal().VAR();                                                 \
        }                                                                                   \
        /** \brief Forwards to l1_proximal **/                                              \
        L1_ADMM<Scalar> & NAME ## _proximal_ ## VAR(                                        \
            decltype(std::declval<proximal::PROXIMAL<Scalar> const>().VAR()) VAR)        {  \
          NAME ## _proximal().VAR(VAR);                                                     \
          return *this;                                                                     \
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
#   undef SOPT_MACRO

    // Includes getters and redefines setters to return this object
#   define SOPT_MACRO(NAME)                                                                 \
        using ADMM<Scalar>::NAME;                                                           \
        /** \brief Forwards to ADMM base class **/                                          \
        L1_ADMM<Scalar> & NAME(decltype(std::declval<ADMM<Scalar>>().NAME()) NAME) {        \
          ADMM<Scalar>::NAME(NAME);                                                         \
          return *this;                                                                     \
        }
    SOPT_MACRO(itermax);
    SOPT_MACRO(gamma);
    SOPT_MACRO(nu);
    SOPT_MACRO(lagrange_update_scale);
    SOPT_MACRO(relative_variation);
    SOPT_MACRO(residual_convergence);
    SOPT_MACRO(Phi);
    SOPT_MACRO(is_converged);
#   undef SOPT_MACRO


    //! Calls l1 proximal operator, checking for real constraints and tight frame
    template<class T0, class T1>
    typename proximal::L1<Scalar>::Diagnostic l1_proximal(
        Eigen::MatrixBase<T0> &out, Real gamma, Eigen::MatrixBase<T1> const &x) const {
      return l1_proximal_real_constraint() ?
        call_l1_proximal(out, gamma, x.real()):
        call_l1_proximal(out, gamma, x);
    }

    //! Forwards call to weighted L2 ball proximal
    template<class T>
      auto l2ball_proximal(Eigen::MatrixBase<T> const& x) const
      -> decltype(std::declval<proximal::WeightedL2Ball<Scalar> const>()(Real(0), x)) {
        return l2ball_proximal()(Real(0), x);
      }


    //! \brief Implements ADMM
    //! \details Follows Combettes and Pesquet "Proximal Splitting Methods in Signal Processing",
    //! arXiv:0912.3522v4 [math.OC] (2010), equation 65.
    //! See therein for notation
    Diagnostic operator()(t_Vector& out, t_Vector const& input) const;

  protected:
    //! Keeps track of the last call to the L1 proximal
    mutable typename proximal::L1<Scalar>::Diagnostic l1_diagnostic;
    //! Calls l1 proximal operator, checking for thight frame
    template<class T0, class T1>
    typename proximal::L1<Scalar>::Diagnostic call_l1_proximal(
        Eigen::MatrixBase<T0> &out, Real gamma, Eigen::MatrixBase<T1> const &x) const {
      if(tight_frame()) {
        l1_proximal().tight_frame(out, gamma, x);
        return {0, 0, l1_proximal().objective(x, out, gamma), true};
      }
      return l1_proximal()(out, gamma, x);
    }

    //! This will be the proximal operator passed to the base class
    void erased_f_proximal(t_Vector &out, Real gamma, t_Vector const &x) const {
      l1_diagnostic = l1_proximal(out, gamma, x);
    }
};

template<class SCALAR>
  typename L1_ADMM<SCALAR>::Diagnostic
  L1_ADMM<SCALAR>::operator()(t_Vector& out, t_Vector const& input) const {

    SOPT_INFO("Performing Proximal ADMM with L1 and L2 operators");
    ADMM<Scalar>::sanity_check(input);

    t_Vector lambda = t_Vector::Zero(input.size()),
             z = t_Vector::Zero(input.size()),
             residual = t_Vector::Zero(input.size());
    bool const has_user_convergence = static_cast<bool>(is_converged());
    l1_diagnostic = {0, 0, 0, false};

    SOPT_NOTICE("    - Initialization");
    ADMM<Scalar>::initialization_step(input, out, residual);
    std::pair<Real, Real> objectives{sopt::l1_norm(residual, l1_proximal_weights()), 0};

    for(t_uint niters(0); niters < itermax(); ++niters) {
      SOPT_NOTICE("    - Iteration {}/{}. ", niters, itermax());
      ADMM<Scalar>::iteration_step(input, out, residual, lambda, z);

      // print-out stuff
      objectives.second = objectives.first;
      objectives.first = sopt::l1_norm(residual, l1_proximal_weights());
      t_real const relative_objective
        = std::abs(objectives.first - objectives.second) / objectives.first;
      auto const residual_norm = sopt::l2_norm(residual, l2ball_proximal_weights());

      SOPT_NOTICE(
          "    - objective: obj value = {}, rel obj = {}", objectives.first, relative_objective);
      SOPT_NOTICE(
          "    - Residuals: epsilon = {}, residual norm = {}",
          l2ball_proximal_epsilon(), residual_norm
      );

      // convergence stuff
      auto const user = (not has_user_convergence) or is_converged(out);
      auto const res = residual_convergence() > 0e0 and residual_norm < residual_convergence();
      auto const rel = relative_variation() > 0e0 and relative_objective < relative_variation();
      if(user and res and rel) {
        SOPT_INFO("    - converged in {} of {} iterations", niters, itermax());
        return Diagnostic{niters, true, l1_diagnostic};
      }
    }

    SOPT_WARN("    - did not converge within {} iterations", itermax());
    return {itermax(), false, l1_diagnostic};
  }

}} /* sopt::algorithm */
#endif
