#ifndef SOPT_CONJUGATE_GRADIENT
#define SOPT_CONJUGATE_GRADIENT

#include "types.h"
#include "utility.h"
#include <iostream>

namespace sopt {
//! Solves 
class ConjugateGradient {
  public:
    //! Values indicating how the algorithm ran
    struct Diagnostic {
      //! Number of iterations
      t_uint niters;
      //! Residual
      t_real residual;
      //! Wether convergence was achieved
      bool good;
    };
    //! Values indicating how the algorithm ran and its result;
    template<class T> struct DiagnosticAndResult : public Diagnostic {
      Eigen::Matrix<T, Eigen::Dynamic, 1> result;
    };
    //! \brief Creates conjugate gradient operator
    //! \param[in] itermax: Maximum number of iterations. 0 means algorithm breaks only if
    //! convergence is reached.
    //! \param[in] tolerance: Convergence criteria
    ConjugateGradient(t_uint itermax=0, t_real tolerance=1e-8)
      : tolerance_(tolerance), itermax_(itermax) {}
    virtual ~ConjugateGradient() {}

    //! Computes $x$ for $Ax=b$
    template<class T0, class T1, class T2>
      Diagnostic operator()(
          Eigen::MatrixBase<T0> &x,
          Eigen::MatrixBase<T1> const &A,
          Eigen::MatrixBase<T2> const &b) const
      {
        return implementation(x, A, b);
      }
    //! Computes $x$ for $Ax=b$
    template<class T0, class T1, class T2>
      Diagnostic operator()(
          Eigen::ArrayBase<T0> &x,
          Eigen::ArrayBase<T1> const &A,
          Eigen::ArrayBase<T2> const &b) const
      {
        return operator()(x.matrix(), A.matrix(), b.matrix());
      }
    //! Computes $x$ for $Ax=b$
    template<class T0, class A_TYPE>
      DiagnosticAndResult<typename T0::Scalar> operator()(
          A_TYPE const& A, Eigen::MatrixBase<T0> const& b) const {
        DiagnosticAndResult<typename T0::Scalar> result;
        result.result = Eigen::Matrix<typename T0::Scalar, Eigen::Dynamic, 1>::Zero(b.size());
        *static_cast<Diagnostic*>(&result) = operator()(result.result, A, b);
        return result;
      }
    template<class T0, class T1>
      DiagnosticAndResult<typename T0::Scalar> operator()(
          Eigen::ArrayBase<T1> const& A, Eigen::ArrayBase<T0> const& b) const {
        return operator()(A.matrix(), b.matrix());
    }

    //! \brief Maximum number of iterations
    //! \details 0 means algorithm breaks only if convergence is reached.
    t_uint itermax() const { return itermax_; }
    //! \brief Sets maximum number of iterations
    //! \details 0 means algorithm breaks only if convergence is reached.
    void itermax(t_uint const &itermax) { itermax_ = itermax; }
    //! Tolerance criteria
    t_real tolerance() const { return tolerance_; }
    //! Sets tolerance criteria
    void tolerance(t_real const &tolerance) {
      if(tolerance <= 0e0)
        throw std::domain_error("Incorrect tolerance input");
      tolerance_ = tolerance;
    }
  protected:
    //! Tolerance criteria
    t_real tolerance_;
    //! Maximum number of iteration
    t_uint itermax_;
    //! \details Work array to hold v
    t_rMatrix work_v;
    //! Work array to hold r
    t_rMatrix work_r;
    //! Work array to hold p
    t_rMatrix work_p;

  private:
    //! \brief Just one implementation for all types
    //! \note This is a template function, to avoid repetition, but it is not declared in the
    //! header.
    template<class T0, class T1, class T2>
      Diagnostic implementation(
        Eigen::MatrixBase<T0> &x, T2 const &A, Eigen::MatrixBase<T1> const & b) const;
};

template<class T0, class T1, class T2>
  ConjugateGradient::Diagnostic ConjugateGradient::implementation(
      Eigen::MatrixBase<T0> &x, T2 const &A, Eigen::MatrixBase<T1> const & b) const {
    typedef typename T0::Scalar Scalar;
    typedef typename underlying_value_type<Scalar>::type Real;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> t_Vector;


    x.resize(b.size());
    if(std::abs((b.transpose().conjugate() * b)(0)) < tolerance()) {
      x.fill(0);
      return {0, 0, 1};
    }

    t_Vector Ap(b.size());
    Scalar alpha;

    x = b;
    t_Vector residuals = b - A*x;
    t_Vector p = residuals;
    Real residual = std::abs((residuals.transpose().conjugate() * residuals)(0));

    t_uint i(0);
    for(; i < itermax() || itermax() == 0 ; ++i) {
      Ap = A * p;
      Scalar alpha = residual / (p.transpose().conjugate() * Ap)(0);
      x += alpha * p;
      residuals -= alpha * Ap;

      Real new_residual = std::abs((residuals.transpose().conjugate() * residuals)(0));
      if(std::abs(new_residual) < tolerance()) {
        residual = new_residual;
        break;
      }

      p = residuals + new_residual / residual * p;
      residual = new_residual;
    }
    return {i, residual, residual < tolerance()};
  }
} /* sopt */ 
#endif
