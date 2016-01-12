#ifndef SOPT_PADMM_H
#define SOPT_PADMM_H

#include <vector>
#include <limits>
#include <numeric>

#include "sopt/types.h"
#include "sopt/linear_transform.h"
#include "sopt/proximal.h"
#include "sopt/wrapper.h"
#include "sopt/exception.h"
#include "sopt/logging.h"
#include "sopt/L1_proximal.h"
#include "sopt/relative_variation.h"

namespace sopt { namespace algorithm {


//! \brief Inexact ADMM-based approach
//! \details \f$\min_{x, z} f(x) + h(z)\f$ subject to \f$Φx + z = y\f$, where \f$f(x) =
//! ||Ψ^Hx||_1 + i_C(x)\f$ and \f$h(x) = i_B(z)\f$ with \f$C = R^N_{+}\f$ and \f$B = {z \in R^M:
//! ||z||_2 \leq \epsilon}\f$.
template<class SCALAR> class PADMM {
  public:
    //! Scalar type
    typedef SCALAR value_type;
    //! Scalar type
    typedef value_type Scalar;
    //! Real type
    typedef typename real_type<Scalar>::type Real;
    //! Type of then underlying vectors
    typedef Vector<SCALAR> t_Vector;
    //! Type of the Ψ and Ψ^H operations, as well as Φ and Φ^H
    typedef LinearTransform<t_Vector> t_LinearTransform;
    //! Type of the convergence function
    typedef ConvergenceFunction<SCALAR> t_IsConverged;

    //! Values indicating how the algorithm ran
    struct Diagnostic {
      //! Number of iterations
      t_uint niters;
      //! Wether convergence was achieved
      bool good;
      //! Diagnostic from calling L1 proximal
      typename proximal::L1<Scalar>::Diagnostic l1_diag;
    };

    PADMM() : itermax_(std::numeric_limits<t_uint>::max()), gamma_(1e-8), nu_(1),
      lagrange_update_scale_(0.9), relative_variation_(1e-4), residual_convergence_(1e-4),
      is_converged_([](t_Vector const&) { return false; }),
      Phi_(linear_transform_identity<Scalar>()),
      target_(t_Vector::Zero(0)), tight_frame_(false),
      l1_proximal_(), weighted_l2ball_proximal_(1e0) {}
    virtual ~PADMM() {}

    // Macro helps define properties that can be initialized as in
    // auto sdmm  = PADMM<float>().prop0(value).prop1(value);
#   define SOPT_MACRO(NAME, TYPE)                                                   \
        TYPE const& NAME() const { return NAME ## _; }                              \
        PADMM<SCALAR> & NAME(TYPE const &NAME) { NAME ## _ = NAME; return *this; }  \
      protected:                                                                    \
        TYPE NAME ## _;                                                             \
      public:
    //! Maximum number of iterations
    SOPT_MACRO(itermax, t_uint);
    //! γ parameter
    SOPT_MACRO(gamma, Real);
    //! ν parameter
    SOPT_MACRO(nu, Real);
    //! Lagrange update scale β
    SOPT_MACRO(lagrange_update_scale, Real);
    //! \brief Convergence of the relative variation of the objective functions
    //! \details If negative, this convergence criteria is disabled.
    SOPT_MACRO(relative_variation, Real);
    //! \brief Convergence of the residuals
    //! \details If negative, this convergence criteria is disabled.
    SOPT_MACRO(residual_convergence, Real);
    //! A function verifying convergence
    //! A function verifying convergence
    SOPT_MACRO(is_converged, t_IsConverged);
    //! Measurement operator
    SOPT_MACRO(Phi, t_LinearTransform);
    //! Target of the measurements
    SOPT_MACRO(target, t_Vector);
    //! Whether Ψ is a tight-frame or not
    SOPT_MACRO(tight_frame, bool);
    //! L1 proximal
    SOPT_MACRO(l1_proximal, proximal::L1<Scalar>);
    //! Proximal of the L2 ball
    SOPT_MACRO(weighted_l2ball_proximal, proximal::WeightedL2Ball<Scalar>);
#   undef SOPT_MACRO
    //! Analysis operator Ψ
    PADMM<Scalar> & Psi(t_LinearTransform const &l) { l1_proximal().Psi(l); return *this; }
    //! \brief Analysis operator Ψ
    //! \details Under-the-hood, the object is actually owned by the L1 proximal.
    t_LinearTransform const & Psi() const { return l1_proximal().Psi(); }
    //! Setup Ψ both here
    //! Ψ initialized via some call to \ref linear_transform
    template<class T0, class ... T>
      typename std::enable_if<
        not std::is_same<typename std::remove_all_extents<T0>::type, t_LinearTransform>::value,
        PADMM<SCALAR> &
      >::type Psi(T0 &&t0, T &&... args) {
        l1_proximal().Psi(linear_transform(std::forward<T0>(t0), std::forward<T>(args)...));
        return *this;
      }
    //! Φ initialized via some call to \ref linear_transform
    template<class T0, class ... T>
      typename std::enable_if<
        not std::is_same<typename std::remove_all_extents<T0>::type, t_LinearTransform>::value,
        PADMM<SCALAR> &
      >::type Phi(T0 &&t0, T &&... args) {
        return Phi(linear_transform(std::forward<T0>(t0), std::forward<T>(args)...));
      }

    //! \brief L1 proximal used during calculation
    //! \details Non-const version to setup the object.
    proximal::L1<Scalar> & l1_proximal() { return l1_proximal_; }
    //! \brief Proximal of the L2 ball
    //! \details Non-const version to setup the object.
    proximal::WeightedL2Ball<Scalar> & weighted_l2ball_proximal() {
      return weighted_l2ball_proximal_;
    }

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
        PADMM<Scalar> & NAME ## _proximal_ ## VAR(                                          \
            decltype(std::declval<proximal::PROXIMAL<Scalar> const>().VAR()) VAR)        {  \
          NAME ## _proximal().VAR(VAR);                                                     \
          return *this;                                                                     \
        }
    SOPT_MACRO(itermax, l1, L1);
    SOPT_MACRO(tolerance, l1, L1);
    SOPT_MACRO(positivity_constraint, l1, L1);
    SOPT_MACRO(real_constraint, l1, L1);
    SOPT_MACRO(fista_mixing, l1, L1);
    SOPT_MACRO(epsilon, weighted_l2ball, WeightedL2Ball);
    SOPT_MACRO(weights, weighted_l2ball, WeightedL2Ball);
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
      auto weighted_l2ball_proximal(Eigen::MatrixBase<T> const& x) const
      -> decltype(std::declval<proximal::WeightedL2Ball<Scalar> const>()(Real(0), x)) {
        return weighted_l2ball_proximal()(Real(0), x);
      }


    //! \brief Implements PADMM
    //! \details Follows Combettes and Pesquet "Proximal Splitting Methods in Signal Processing",
    //! arXiv:0912.3522v4 [math.OC] (2010), equation 65.
    //! See therein for notation
    Diagnostic operator()(t_Vector& out, t_Vector const& input) const;

    bool relative_variation_convergence(Real previous, Real current) const {
      return relative_variation() > 0e0
        and std::abs(previous - current) > std::abs(current) * relative_variation();
    }
    bool residual_norm_convergence(Real residual) const {
      return residual_convergence() > 0e0 and residual < residual_convergence();
    }

  protected:
    //! Checks convergence
    //! \param[in] x: current solution
    //! \param[in] previous: previous objective function
    //! \param[in] current: current objective function
    //! \param[in] residual: norm of the residuals
    bool is_converged(t_Vector const &x, Real previous, Real current, Real residual) const {
      return relative_variation_convergence(previous, current)
        and residual_norm_convergence(residual)
        and is_converged()(x);
    }

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

};

template<class SCALAR>
  typename PADMM<SCALAR>::Diagnostic
  PADMM<SCALAR>::operator()(t_Vector& out, t_Vector const& input) const {

    if((Phi().adjoint() * target()).size() != input.size()) {
      SOPT_THROW("target, measurement operator and input vector have inconsistent sizes");
    }

    SOPT_INFO("Performing approximate PADMM ");
    SOPT_NOTICE("- Initialization");
    out = Phi().adjoint() * target() / nu();
    t_Vector residual = Phi() * out - target();
    Real objective = sopt::l1_norm(Psi().adjoint() * out);
    typename proximal::L1<Scalar>::Diagnostic l1_diag{0, 0, 0, false};

    t_Vector lambda = t_Vector::Zero(target().size());
    for(t_uint niters(0); niters < itermax(); ++niters) {
      SOPT_NOTICE("Iteration {}/{}. ", niters, itermax());

      // Iteration code
      t_Vector const z = weighted_l2ball_proximal(-lambda - residual);
      l1_diag = l1_proximal(
          out, gamma() / nu(), out - Phi().adjoint() * (residual + lambda + z) / nu());
      residual = Phi() * out - target();
      lambda += lagrange_update_scale() * (residual + z);

      // Print-out stuff
      auto const previous_objective = objective;
      objective = sopt::l1_norm(Psi().adjoint() * out);
      t_real const relative_objective = std::abs(previous_objective - objective) / objective;
      t_real const norm_residuals = (
          residual.array() * weighted_l2ball_proximal_weights().array()).matrix().stableNorm();
      SOPT_NOTICE(
          "    - objective: obj value = {}, rel obj = {}\n"
          "    - Residuals: epsilon = {}, residual norm = {}",
          objective, relative_objective, weighted_l2ball_proximal_epsilon(), norm_residuals
      );

      // Convergence checking
      if(is_converged(out, previous_objective, objective, norm_residuals)) {
        SOPT_INFO("Approximate PADDM converged in {} of {} iterations", niters, itermax());
        return {niters, true, l1_diag};
      }
    }

    SOPT_WARN("Approximate PADDM did not converge within {} iterations", itermax());
    return {itermax(), false, l1_diag};
  }

}} /* sopt::algorithm */
#endif
