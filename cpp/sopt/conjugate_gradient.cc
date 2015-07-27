#include "conjugate_gradient.h"

namespace sopt {

template<class T0, class T1, class T2>
  ConjugateGradient::Diagnostic ConjugateGradient::implementation(
      Eigen::ArrayBase<T0> &out, Eigen::ArrayBase<T1> const & input, T2 const &A) const {

    out.resize(input.size());
    if((input * input).sum() < tolerance()) {
      out.fill(0);
      return {0, 0, 1};
    }

    typedef typename T0::Scalar Scalar;
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> t_Vector;

    t_Vector Ap(input.size());
    Scalar alpha;

    auto is_converged = [this, &input](Scalar const &residual) {
      return residual < this->tolerance() * input.size();
    };

    t_Vector residuals = input - A*input;
    t_Vector p = residuals;
    auto residual = (residuals * residuals).real().sum();

    t_uint i(0);
    for(; i < itermax() and is_converged(residual); ++i) {
      Ap.noalias() = A * p;
      auto const alpha = residual / (Ap * p).real().sum();
      out += alpha * p;
      residuals -= Ap;
      auto const new_residual = (residuals * residuals).creal().sum();
      p = residuals + new_residual / residual * p;
      residual = new_residual;
    }
    return {i, residual, is_converged(residual)};
  }

# define SOPT_MACRO(TYPE)                                                               \
      ConjugateGradient::Diagnostic ConjugateGradient::operator()(                      \
            TYPE &out, TYPE input,                                                      \
            std::function<void(TYPE&, t_rVector const&)> const &applyA) const {         \
        auto Ax_plus_2x = [&applyA](TYPE& out, TYPE const&input) {                      \
          applyA(out, input);                                                           \
          out += 2*input;                                                               \
        };                                                                              \
        return implementation(out, input, Ax_plus_2x);                                  \
      }
  SOPT_MACRO(t_rVector);
  SOPT_MACRO(t_cVector);
  SOPT_MACRO(t_rMatrix);
  SOPT_MACRO(t_cMatrix);
# undef SOPT_MACRO

} /* sopt */
