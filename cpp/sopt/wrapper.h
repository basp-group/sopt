#ifndef SOPT_WRAP
#define SOPT_WRAP

#include <type_traits>
#include "types.h"
#include "utility.h"
#include <iostream>

namespace sopt { namespace details {
//! Expression referencing the result of a function call
template<class DERIVED>
  class AppliedFunction : public Eigen::ReturnByValue<AppliedFunction<DERIVED>>{
    public:
      typedef typename DERIVED::PlainObject PlainObject;
      typedef typename DERIVED::Index Index;
      typedef std::function<void(PlainObject &out, PlainObject const &input)> t_Function;

      AppliedFunction(t_Function const &func, DERIVED const &x) : func(func), x(x) {}

      template<class DESTINATION> void evalTo(DESTINATION &destination) const {
        return func(destination, x);
      }

      Index rows() const { return x.rows(); }
      Index cols() const { return x.cols(); }

    private:
      t_Function const &func;
      DERIVED const &x;
  };

//! \brief Wraps an std::function to return an expression
//! \details This makes writing the application of a function more beautiful on the eye.
//! A function call `func(output, input)` can be made to look like `output = func(input)` or `output
//! = func * input`.
template<class VECTOR> class WrapFunction {
  public:
    typedef std::function<void(VECTOR &out, VECTOR const &input)> t_Function;

    WrapFunction(t_Function const &func) : func(func) {}

    //! Function application form
    template<class T0>
      AppliedFunction<Eigen::ArrayBase<T0>> operator()(Eigen::ArrayBase<T0> const &input) const {
        return AppliedFunction<Eigen::ArrayBase<T0>>(func, input);
      }

    //! Multiplication application form
    template<class T0>
      AppliedFunction<Eigen::ArrayBase<T0>> operator*(Eigen::ArrayBase<T0> const &input) const {
        return AppliedFunction<Eigen::ArrayBase<T0>>(func, input);
      }

    //! Function application form
    template<class T0>
      AppliedFunction<Eigen::MatrixBase<T0>> operator()(Eigen::MatrixBase<T0> const &input) const {
        return AppliedFunction<Eigen::MatrixBase<T0>>(func, input);
      }

    //! Multiplication application form
    template<class T0>
      AppliedFunction<Eigen::MatrixBase<T0>> operator*(Eigen::MatrixBase<T0> const &input) const {
        return AppliedFunction<Eigen::MatrixBase<T0>>(func, input);
      }

  private:
    //! Reference function
    t_Function const &func;
};

template<class VECTOR>
  WrapFunction<VECTOR> wrap(std::function<void(VECTOR &input, VECTOR const& out)> const &func) {
    return WrapFunction<VECTOR>(func);
  }

}}

namespace Eigen { namespace internal {
  template<typename VECTOR> struct traits<sopt::details::AppliedFunction<VECTOR>> {
    typedef typename VECTOR::PlainObject ReturnType;
  };

}}
#endif
