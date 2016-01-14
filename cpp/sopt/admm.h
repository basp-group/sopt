#ifndef SOPT_ADMM_H
#define SOPT_ADMM_H

#include <limits>
#include <functional>

#include "sopt/types.h"
#include "sopt/linear_transform.h"
#include "sopt/exception.h"
#include "sopt/logging.h"

namespace sopt { namespace algorithm {


//! \brief Alternate Direction method of mutltipliers
//! \details \f$\min_{x, z} f(x) + h(z)\f$ subject to \f$Φx + z = y\f$.
template<class SCALAR> class ADMM {
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
      Diagnostic(t_uint niters, bool good) : niters(niters), good(good) {}
    };

    //! Setups ADMM
    //! \param[in] f_proximal: proximal operator of the \f$f\f$ function.
    //! \param[in] g_proximal: proximal operator of the \f$g\f$ function
    ADMM(t_Proximal const & f_proximal, t_Proximal const & g_proximal)
      : itermax_(std::numeric_limits<t_uint>::max()), gamma_(1e-8), nu_(1),
        lagrange_update_scale_(0.9), relative_variation_(1e-4), residual_convergence_(1e-4),
        is_converged_(), Phi_(linear_transform_identity<Scalar>()),
        f_proximal_(f_proximal), g_proximal_(g_proximal) {}
    virtual ~ADMM() {}

    // Macro helps define properties that can be initialized as in
    // auto sdmm  = ADMM<float>().prop0(value).prop1(value);
#   define SOPT_MACRO(NAME, TYPE)                                                   \
        TYPE const& NAME() const { return NAME ## _; }                              \
        ADMM<SCALAR> & NAME(TYPE const &NAME) { NAME ## _ = NAME; return *this; }  \
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
    //! First proximal
    SOPT_MACRO(f_proximal, t_Proximal);
    //! Second proximal
    SOPT_MACRO(g_proximal, t_Proximal);
#   undef SOPT_MACRO
    //! \brief Simplifies calling the proximal of f.
    void f_proximal(t_Vector & out, Real gamma, t_Vector const &x) const {
      f_proximal()(out, gamma, x);
    }
    //! \brief Simplifies calling the proximal of f.
    void g_proximal(t_Vector & out, Real gamma, t_Vector const &x) const {
      g_proximal()(out, gamma, x);
    }

    //! Facilitates call to user-provided convergence function
    bool is_converged(t_Vector const &x) const {
      return is_converged() and is_converged()(x);
    }

    //! \brief Implements ADMM
    //! \details Follows Combettes and Pesquet "Proximal Splitting Methods in Signal Processing",
    //! arXiv:0912.3522v4 [math.OC] (2010), equation 65.
    //! See therein for notation
    Diagnostic operator()(t_Vector& out, t_Vector const& input) const;

  protected:
    void initialization_step(t_Vector const& input, t_Vector &out, t_Vector &residual) const;
    void iteration_step(
        t_Vector const &input, t_Vector &out,
        t_Vector &residual, t_Vector &lambda, t_Vector &z) const;

    //! Checks input makes sense
    void sanity_check(t_Vector const&) const {
      // if((Phi().adjoint() * target()).size() != input.size())
      //   SOPT_THROW("target, measurement operator and input vector have inconsistent sizes");
      if(not static_cast<bool>(is_converged()))
        SOPT_WARN(
            "No convergence function was provided: algorithm will run for {} steps", itermax());
    }
};

template<class SCALAR>
  void ADMM<SCALAR>::initialization_step(
      t_Vector const & input, t_Vector& out, t_Vector & residual) const {
    out = Phi().adjoint() * input / nu();
    residual = Phi() * out - input;
  }
template<class SCALAR>
  void ADMM<SCALAR>::iteration_step(
      t_Vector const &input, t_Vector &out, t_Vector &residual,
      t_Vector &lambda, t_Vector &z) const
{
    g_proximal(z, gamma(), -lambda - residual);
    f_proximal(out, gamma() / nu(), out - Phi().adjoint() * (residual + lambda + z) / nu());
    residual = Phi() * out - input;
    lambda += lagrange_update_scale() * (residual + z);
  }

template<class SCALAR>
  typename ADMM<SCALAR>::Diagnostic
  ADMM<SCALAR>::operator()(t_Vector& out, t_Vector const& input) const {

    SOPT_INFO("Performing approximate Proximal ADMM");
    sanity_check(input);

    t_Vector lambda = t_Vector::Zero(input.size()),
             z = t_Vector::Zero(input.size()),
             residual = t_Vector::Zero(input.size());

    SOPT_NOTICE("    - Initialization");
    initialization_step(out, residual);

    for(t_uint niters(0); niters < itermax(); ++niters) {
      SOPT_NOTICE("    - Iteration {}/{}. ", niters, itermax());
      iteration_step(input, out, residual, lambda, z);

      if(static_cast<bool>(is_converged()) and is_converged(out)) {
        SOPT_INFO("    - converged in {} of {} iterations", niters, itermax());
        return {niters, true};
      }
    }
    // check function exists, otherwise, don't know if convergence is meaningfull
    if(static_cast<bool>(is_converged()))
      SOPT_WARN("    - did not converge within {} iterations", itermax());
    return {itermax(), not is_converged()};
  }

}} /* sopt::algorithm */
#endif
