#ifndef SOPT_RELATIVE_VARIATION_H
#define SOPT_RELATIVE_VARIATION_H

#include <Eigen/Core>

#include "sopt/utility.h"

namespace sopt {

  template<class TYPE> class RelativeVariation {
    public:
      //! Underlying scalar type
      typedef TYPE Scalar;
      //! Underlying scalar type
      typedef typename real_type<Scalar>::type Real;
      //! Maximum variation from one step to the next
      RelativeVariation(Real epsilon) : epsilon_(epsilon), is_first(true) {};

      //! True if object has changed by less than epsilon
      template<class T>
        bool operator()(Eigen::MatrixBase<T> const &input) {
          return operator()(input.array());
        }
      //! True if object has changed by less than epsilon
      template<class T>
        bool operator()(Eigen::ArrayBase<T> const &input) {
          if(is_first) {
            is_first = false;
            previous = input;
            return false;
          }
          auto const result = (input - previous).matrix().squaredNorm() < epsilon() * epsilon();
          previous = input;
          return result;
        }
      //! Allowed variation
      Real epsilon() const { return epsilon_; }
      //! Allowed variation
      RelativeVariation& epsilon(Real &e) const { epsilon_ = e; return *this; }

    protected:
      typename real_type<Scalar>::type epsilon_;
      bool is_first;
      Eigen::Array<Scalar, Eigen::Dynamic, 1> previous;
  };
} /* sopt  */

#endif
