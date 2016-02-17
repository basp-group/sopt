#ifndef SOPT_WRAP
#define SOPT_WRAP

#include <array>
#include <type_traits>

#include "sopt/config.h"
#include "sopt/exception.h"
#include "sopt/types.h"
#include "sopt/utility.h"

namespace sopt {
namespace details {
//! Expression referencing the result of a function call
template <class FUNCTION, class DERIVED>
class AppliedFunction : public Eigen::ReturnByValue<AppliedFunction<FUNCTION, DERIVED>> {
public:
  typedef typename DERIVED::PlainObject PlainObject;
  typedef typename DERIVED::Index Index;

  AppliedFunction(FUNCTION const &func, DERIVED const &x, Index rows)
      : func(func), x(x), rows_(rows) {}
  AppliedFunction(FUNCTION const &func, DERIVED const &x) : func(func), x(x), rows_(x.rows()) {}
  AppliedFunction(AppliedFunction const &c) : func(c.func), x(c.x), rows_(c.rows_) {}
  AppliedFunction(AppliedFunction &&c) : func(std::move(c.func)), x(c.x), rows_(c.rows_) {}

  template <class DESTINATION> void evalTo(DESTINATION &destination) const { func(destination, x); }

  Index rows() const { return rows_; }
  Index cols() const { return x.cols(); }

private:
  FUNCTION const func;
  DERIVED const &x;
  Index const rows_;
};

//! \brief Wraps an std::function to return an expression
//! \details This makes writing the application of a function more beautiful on the eye.
//! A function call `func(output, input)` can be made to look like `output = func(input)` or `output
//! = func * input`.
template <class VECTOR> class WrapFunction {
public:
  //! Type of function wrapped here
  typedef OperatorFunction<VECTOR> t_Function;

  //! Initializes the wrapper
  //! \param[in] func: function to wrap
  //! \param[in] sizes: three integer vector (a, b, c)
  //! if N is the size of the input, then (N * a) / b  + c is the output
  //! b cannot be zero.
  WrapFunction(t_Function const &func, std::array<t_int, 3> sizes = {{1, 1, 0}})
      : func(func), sizes_(sizes) {
    // cannot devide by zero
    assert(sizes_[1] != 0);
  }
  WrapFunction(WrapFunction const &c) : func(c.func), sizes_(c.sizes_) {}
  WrapFunction(WrapFunction const &&c) : func(std::move(c.func)), sizes_(std::move(c.sizes_)) {}
  void operator=(WrapFunction const &c) {
    func = c.func;
    sizes_ = c.sizes_;
  }
  void operator=(WrapFunction &&c) {
    func = std::move(c.func);
    sizes_ = std::move(c.sizes_);
  }

  //! Function application form
  template <class T0>
  AppliedFunction<t_Function const &, Eigen::ArrayBase<T0>>
  operator()(Eigen::ArrayBase<T0> const &x) const {
    return AppliedFunction<t_Function const &, Eigen::ArrayBase<T0>>(func, x, rows(x));
  }

  //! Multiplication application form
  template <class T0>
  AppliedFunction<t_Function const &, Eigen::ArrayBase<T0>>
  operator*(Eigen::ArrayBase<T0> const &x) const {
    return AppliedFunction<t_Function const &, Eigen::ArrayBase<T0>>(func, x, rows(x));
  }

  //! Function application form
  template <class T0>
  AppliedFunction<t_Function const &, Eigen::MatrixBase<T0>>
  operator()(Eigen::MatrixBase<T0> const &x) const {
    return AppliedFunction<t_Function const &, Eigen::MatrixBase<T0>>(func, x, rows(x));
  }

  //! Multiplication application form
  template <class T0>
  AppliedFunction<t_Function const &, Eigen::MatrixBase<T0>>
  operator*(Eigen::MatrixBase<T0> const &x) const {
    return AppliedFunction<t_Function const &, Eigen::MatrixBase<T0>>(func, x, rows(x));
  }

  //! \brief Defines relation-ship between input and output sizes
  //! \details An integer tuple (a, b, c) where, if N is the size of the input, then
  //! \f$(N * a) / b + c\f$ is the output. \f$b\f$ cannot be zero.
  //! In the simplest case where this objects wraps a square matrix, then the sizes are (1, 1, 0).
  //! If this objects wraps a rectangular matrix which halves the number of elements, then the
  //! sizes would be (1, 2, 0).
  std::array<t_int, 3> const &sizes() const { return sizes_; }

  //! Output vector size for a input with `xsize` elements
  template <class T>
  typename std::enable_if<std::is_integral<T>::value, T>::type rows(T xsize) const {
    auto const result = (static_cast<t_int>(xsize) * sizes_[0]) / sizes_[1] + sizes_[2];
    assert(result > 0);
    return static_cast<T>(result);
  }

protected:
  template <class T> t_uint rows(Eigen::DenseBase<T> const &x) const { return rows(x.size()); }

private:
  //! Reference function
  t_Function func;
  //! Ratio between input and output size
  std::array<t_int, 3> sizes_;
};

//! Helper function to wrap functor into expression-able object
template <class VECTOR>
WrapFunction<VECTOR>
wrap(OperatorFunction<VECTOR> const &func, std::array<t_int, 3> sizes = {{1, 1, 0}}) {
  return WrapFunction<VECTOR>(func, sizes);
}
}
}

namespace Eigen {
namespace internal {
template <class FUNCTION, class VECTOR>
struct traits<sopt::details::AppliedFunction<FUNCTION, VECTOR>> {
  typedef typename VECTOR::PlainObject ReturnType;
};
}
}
#endif
