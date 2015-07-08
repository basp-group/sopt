#ifndef SOPT_CONJUGATE_GRADIENT
#define SOPT_CONJUGATE_GRADIENT

#include <type_traits>
#include "types.h"
#include "utility.h"
#include <iostream>

namespace sopt {
//! Solves $Ax = b$ for $x$, given $A$ and $b$.
class ConjugateGradient {
  //! \brief Wraps around a matrix to fake a functor
  //! \details xout = A * xin becomes apply_matrix_instance(xout, xin);
  template<class T> class ApplyMatrix;
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

    //! \brief Computes $x$ for $Ax=b$
    //! \details Specialization that converts A from a matrix to a functor.
    //! This convertion is only so we write the conjugate-gradient algorithm only once for
    //! A as a matrix and A as a functor. A as a functor means A can be a complex operation, e.g. an
    //! FFT or two.
    template<class T0, class T1, class T2>
      Diagnostic operator()(
          Eigen::MatrixBase<T0> &x,
          Eigen::MatrixBase<T1> const &A,
          Eigen::MatrixBase<T2> const &b) const {
        return operator()(x, ApplyMatrix<decltype(A)>(A), b);
      }
    //! \brief Computes $x$ for $Ax=b$
    //! \details Specialization that converts x and b from array-like object to matrix-like objects.
    template<class T0, class T1, class T2>
      Diagnostic operator()(
          Eigen::ArrayBase<T0> &x,
          Eigen::ArrayBase<T1> const &A,
          Eigen::ArrayBase<T2> const &b) const {
        return operator()(x.matrix(), A.matrix(), b.matrix());
      }
    //! \brief Computes $x$ for $Ax=b$
    //! \details Specialization that converts x and b from array-like object to matrix-like objects.
    //! It expects A to be a functor.
    template<class T0, class T1, class T2>
      typename std::enable_if<
        not std::is_base_of<Eigen::EigenBase<T1>, T1>::value,
        Diagnostic
      >::type operator()(
          Eigen::ArrayBase<T0> &x,
          T1 const &A,
          Eigen::ArrayBase<T2> const &b) const {
        return operator()(x.matrix(), A, b.matrix());
      }
    //! \brief Computes $x$ for $Ax=b$
    //! \details Specialisation where A is a functor and b and x are matrix-like objects. This is
    //! the innermost specialization.
    template<class T0, class T1, class T2>
      typename std::enable_if<
        not std::is_base_of<Eigen::EigenBase<T1>, T1>::value,
        Diagnostic
      >::type operator()(
          Eigen::MatrixBase<T0> &x,
          T1 const &A,
          Eigen::MatrixBase<T2> const &b) const {
        return implementation(x, A, b);
      }
    //! \brief Computes $x$ for $Ax=b$
    //! \details Specialisation where x is constructed during call and returned.
    template<class T0, class A_TYPE>
      DiagnosticAndResult<typename T0::Scalar> operator()(
          A_TYPE const& A, Eigen::MatrixBase<T0> const& b) const {
        DiagnosticAndResult<typename T0::Scalar> result;
        result.result = Eigen::Matrix<typename T0::Scalar, Eigen::Dynamic, 1>::Zero(b.size());
        *static_cast<Diagnostic*>(&result) = operator()(result.result, A, b);
        return result;
      }
    //! \brief Computes $x$ for $Ax=b$
    //! \details Specialisation where x is constructed during call and returned.
    template<class T0, class T1>
      DiagnosticAndResult<typename T0::Scalar> operator()(
          Eigen::ArrayBase<T1> const& A, Eigen::ArrayBase<T0> const& b) const {
        return operator()(A.matrix(), b.matrix());
    }

    template<class T0, class T_FUNC>
      Diagnostic operator()(Eigen::ArrayBase<T0> &x, T_FUNC A, Eigen::ArrayBase<T0> const &b) const;

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

template<class T> class ConjugateGradient :: ApplyMatrix {
  public:
    ApplyMatrix(T const & A) : A(A) {};
    template<class T0, class T1> void operator()(
        Eigen::MatrixBase<T0> &out, Eigen::MatrixBase<T1> const &input) const {
      out = A * input;
    }
  private:
    T const &A;
};

template<class T0, class T1, class T2>
  ConjugateGradient::Diagnostic ConjugateGradient::implementation(
      Eigen::MatrixBase<T0> &x, T2 const &applyA, Eigen::MatrixBase<T1> const & b) const {
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
    t_Vector residuals(b.size());
    applyA(residuals, x);
    residuals = b - residuals;
    t_Vector p = residuals;
    Real residual = std::abs((residuals.transpose().conjugate() * residuals)(0));

    t_uint i(0);
    for(; i < itermax() || itermax() == 0 ; ++i) {
      applyA(Ap, p);
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
