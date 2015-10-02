#ifndef SOPT_RELATIVE_VARIATION_H
#define SOPT_RELATIVE_VARIATION_H

#include <Eigen/Core>

#include "sopt/utility.h"
#include "sopt/logging.h"

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
          auto const norm = (input - previous).matrix().squaredNorm();
          previous = input;
          SOPT_DEBUG("    - relative variation: {} <? {}", std::sqrt(norm), epsilon());
          return norm < epsilon() * epsilon();
        }
      //! Allowed variation
      Real epsilon() const { return epsilon_; }
      //! Allowed variation
      RelativeVariation& epsilon(Real &e) const { epsilon_ = e; return *this; }

    protected:
      typename real_type<Scalar>::type epsilon_;
      bool is_first;
      Array<Scalar> previous;
  };
} /* sopt  */

#endif
