#ifndef SOPT_L1_H
#define SOPT_L1_H

#include <vector>
#include <limits>

#include "types.h"
#include "linear_transform.h"
#include "proximals.h"
#include "wrapper.h"
#include "exception.h"
#include "conjugate_gradient.h"

namespace sopt {

template<class SCALAR> class L1SDMM {
  public:
    //! Values indicating how the algorithm ran
    struct Diagnostic {
      //! Number of iterations
      t_uint niters;
      //! Wether convergence was achieved
      bool good;
      //! Conjugate gradient result
      ConjugateGradient::Diagnostic cg_diagnostic;
    };
    //! Scalar type
    typedef SCALAR value_type;
    //! Scalar type
    typedef value_type Scalar;
    //! Type of then underlying vectors
    typedef Eigen::Matrix<SCALAR, Eigen::Dynamic, 1> t_Vector;
    //! Type of the A and A^t operations
    typedef LinearTransform<t_Vector> t_LinearTransform;
    //! Type of the proximal functions
    typedef std::function<void(t_Vector &, Scalar, t_Vector const &)> t_Proximal;
    //! Type of the convergence function
    typedef std::function<bool(L1SDMM const&, t_Vector const&)> t_IsConverged;

    L1SDMM() : itermax_(std::numeric_limits<t_uint>::max()), gamma_(1e-8),
      conjugate_gradient_(std::numeric_limits<t_uint>::max(), 1e-6),
      is_converged_([](L1SDMM const&, t_Vector const&) { return false; }) {}
    virtual ~L1SDMM() {}

    // Macro helps define properties that can be initialized as in
    // auto sdmm  = L1SDMM<float>().prop0(value).prop1(value);
#   define SOPT_MACRO(NAME, TYPE)                                                   \
        TYPE NAME() const { return NAME ## _; }                                     \
        L1SDMM<SCALAR> & NAME(TYPE const &NAME) { NAME ## _ = NAME; return *this; } \
      protected:                                                                    \
        TYPE NAME ## _;                                                             \
      public:
    //! Maximum number of iterations
    SOPT_MACRO(itermax, t_uint);
    //! Gamma
    SOPT_MACRO(gamma, Scalar);
    //! Conjugate gradient
    SOPT_MACRO(conjugate_gradient, ConjugateGradient);
    //! A function verifying convergence
    SOPT_MACRO(is_converged, t_IsConverged);
#   undef SOPT_MACRO
    //! Helps setup conjugate gradient
    L1SDMM<SCALAR> & conjugate_gradient(t_uint itermax, t_real tolerance) {
      conjugate_gradient().itermax(itermax);
      conjugate_gradient().tolerance(tolerance);
      return *this;
    }
    //! \brief Appends a proximal and linear transform
    //! \details Makes for easier construction
    template<class PROXIMAL, class ... T> L1SDMM<SCALAR>& append(PROXIMAL proximal, T ... args) {
      proximals().emplace_back(proximal);
      transforms().emplace_back(linear_transform(args...));
      return *this;
    }

    //! \brief Implements SDMM
    //! \details Follows Combettes and Pesquet "Proximal Splitting Methods in Signal Processing",
    //! arXiv:0912.3522v4 [math.OC] (2010), equation 65.
    //! See therein for notation
    Diagnostic operator()(t_Vector& out, t_Vector const& input) const;

    //! Linear transforms associated with each objective function
    std::vector<t_LinearTransform> const & transforms() const { return transforms_; }
    //! Linear transforms associated with each objective function
    std::vector<t_LinearTransform> & transforms() { return transforms_; }
    //! Linear transform associated with a given objective function
    t_LinearTransform const & transforms(t_uint i) const { return transforms_[i]; }

    //! Proximal of each objective function
    std::vector<t_Proximal> const & proximals() const { return proximals_; }
    //! Linear transforms associated with each objective function
    std::vector<t_Proximal> & proximals() { return proximals_; }
    //! Lazy call to specific proximal function
    template<class T0>
      proximals::details::AppliedProximalFunction<t_Proximal const&, Eigen::MatrixBase<T0>>
      proximals(t_uint i, Eigen::MatrixBase<T0> const &x) const {
        typedef proximals::details::AppliedProximalFunction<
          t_Proximal const&, Eigen::MatrixBase<T0>
        > t_LazyProximal;
        return t_LazyProximal(proximals()[i], gamma(), x);
      }

    //! \brief Forwards to internal conjugage gradient object
    //! \details Removes the need for ugly extra brackets.
    template<class ... T>
      auto conjugate_gradient(T& ... args) const -> decltype(this->conjugate_gradient()(args...)) {
        return conjugate_gradient()(args...);
      }

    //! Forwards to convergence function parameter
    bool is_converged(t_Vector const &x) const { return is_converged()(*this, x); }

  protected:
    //! Linear transforms associated with each objective function
    std::vector<t_LinearTransform> transforms_;
    //! Proximal of each objective function
    std::vector<t_Proximal> proximals_;

    //! Type of the list of vectors
    typedef std::vector<t_Vector> t_Vectors;
    //! Conjugate gradient step
    virtual ConjugateGradient::Diagnostic solve_for_xn(
        t_Vector &out, t_Vectors const &y, t_Vectors const& z) const;
    //! Direction step
    virtual void update_directions(t_Vectors& y, t_Vectors& z, t_Vector const& x) const;

    //! Initializes intermediate values
    virtual void initialization(t_Vectors &y, t_Vector const &x) const;
};

template<class SCALAR>
  typename L1SDMM<SCALAR>::Diagnostic
  L1SDMM<SCALAR>::operator()(t_Vector& out, t_Vector const& input) const {
    ConjugateGradient::Diagnostic cg_diagnostic;
    bool  convergence = false;
    t_uint niters (0);
    // Figures out where itermax or convergence reached
    auto const has_finished = [&convergence, &niters, this](t_Vector const&out) {
      convergence = is_converged(out);
      return niters >= itermax() or convergence;
    };

    out = input;
    t_Vectors y(transforms().size(), t_Vector::Zero(out.size()));
    t_Vectors z(transforms().size(), t_Vector::Zero(out.size()));

    // Initial step replaces iteration update with initialization
    initialization(y, input);
    cg_diagnostic = solve_for_xn(out, y, z);

    while(not has_finished(out)) {
      // computes y and z from out and transforms
      update_directions(y, z, out);
      // computes x = L^-1 y
      cg_diagnostic = solve_for_xn(out, y, z);

      ++niters;
    }
    return {niters, convergence, cg_diagnostic};
  }

template<class SCALAR>
  ConjugateGradient::Diagnostic L1SDMM<SCALAR>::solve_for_xn(
      t_Vector &out, t_Vectors const &y, t_Vectors const &z) const {

    assert(z.size() == transforms().size());
    assert(y.size() == transforms().size());

    // Initialize b of A x = b = sum_i L_i^T(z_i - y_i)
    t_Vector b = out.Zero(out.size());
    for(t_uint i(0); i < transforms().size(); ++i)
      b += (transforms(i).dagger() * (y[i] - z[i]));

    // Then create operator A
    auto A = [this](t_Vector &out, t_Vector const &input) {
      out = out.Zero(input.size());
      for(auto const &transform: this->transforms()) {
        t_Vector const a = transform * input;
        t_Vector const LtLb_i = transform.dagger() * a;
        out += LtLb_i;
      }
    };

    // Call conjugate gradient
    auto const diagnostic = this->conjugate_gradient(out, A, b);
    if(not diagnostic.good)
      SOPT_THROW("CG error:\n")
        << "  - itermax: " << conjugate_gradient().itermax() << "\n"
        << "  - iter: " << diagnostic.niters << "\n"
        << "  - residuals: " << diagnostic.residual << "\n";

    return diagnostic;
  }

template<class SCALAR>
  void L1SDMM<SCALAR>::update_directions(t_Vectors& y, t_Vectors& z, t_Vector const& x) const {
    for(t_uint i(0); i < transforms().size(); ++i) {
      t_Vector const Lx_i = transforms(i) * x;
      z[i] += Lx_i;
      y[i] = proximals(i, z[i]);
      z[i] -= y[i];
    }
  }

template<class SCALAR>
  void L1SDMM<SCALAR>::initialization(t_Vectors& y, t_Vector const& x) const {
    for(t_uint i(0); i < transforms().size(); i++)
      y[i] = transforms(i) * x;
  }
} /* sopt */
#endif
