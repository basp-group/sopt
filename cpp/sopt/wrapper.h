#ifndef SOPT_WRAP
#define SOPT_WRAP

#include <type_traits>
#include "types.h"
#include "utility.h"
#include <iostream>

namespace sopt { namespace details {
//! Expression referencing the result of a function call
template<class FUNCTION, class DERIVED>
  class AppliedFunction : public Eigen::ReturnByValue<AppliedFunction<FUNCTION, DERIVED>>{
    public:
      typedef typename DERIVED::PlainObject PlainObject;
      typedef typename DERIVED::Index Index;

      AppliedFunction(FUNCTION const &func, DERIVED const &x) : func(func), x(x) {}

      template<class DESTINATION> void evalTo(DESTINATION &destination) const {
        destination.resizeLike(x);
        assert(func);
        func(destination, x);
      }

      Index rows() const { return x.rows(); }
      Index cols() const { return x.cols(); }

    private:
      FUNCTION const func;
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
      AppliedFunction<t_Function const &, Eigen::ArrayBase<T0>> operator()(
          Eigen::ArrayBase<T0> const &input) const {
        return AppliedFunction<t_Function const&, Eigen::ArrayBase<T0>>(func, input);
      }

    //! Multiplication application form
    template<class T0>
      AppliedFunction<t_Function const &, Eigen::ArrayBase<T0>> operator*(
          Eigen::ArrayBase<T0> const &input) const {
        return AppliedFunction<t_Function const&, Eigen::ArrayBase<T0>>(func, input);
      }

    //! Function application form
    template<class T0>
      AppliedFunction<t_Function const &, Eigen::MatrixBase<T0>> operator()(
          Eigen::MatrixBase<T0> const &input) const {
        return AppliedFunction<t_Function const &, Eigen::MatrixBase<T0>>(func, input);
      }

    //! Multiplication application form
    template<class T0>
      AppliedFunction<t_Function const &, Eigen::MatrixBase<T0>> operator*(
          Eigen::MatrixBase<T0> const &input) const {
        return AppliedFunction<t_Function const &, Eigen::MatrixBase<T0>>(func, input);
      }

  private:
    //! Reference function
    t_Function const func;
};

//! Helper function to wrap functor into expression-able object
template<class VECTOR>
  WrapFunction<VECTOR> wrap(std::function<void(VECTOR &input, VECTOR const& out)> const &func) {
    return WrapFunction<VECTOR>(func);
  }

}}

namespace Eigen { namespace internal {
  template<class FUNCTION, class VECTOR>
    struct traits<sopt::details::AppliedFunction<FUNCTION, VECTOR>> {
      typedef typename VECTOR::PlainObject ReturnType;
    };

}}
#endif
