#ifndef SOPT_RELATIVE_VARIATION_H
#define SOPT_RELATIVE_VARIATION_H

#include "sopt/config.h"
#include <Eigen/Core>
#include "sopt/logging.h"
#include "sopt/maths.h"

namespace sopt {

template <class TYPE> class RelativeVariation {
public:
  //! Underlying scalar type
  typedef TYPE Scalar;
  //! Underlying scalar type
  typedef typename real_type<Scalar>::type Real;
  //! Maximum variation from one step to the next
  RelativeVariation(Real epsilon) : epsilon_(epsilon), previous(typename Array<Scalar>::Index(0)){};
  //! Copy constructor
  RelativeVariation(RelativeVariation const &c) : epsilon_(c.epsilon_), previous(c.previous){};

  //! True if object has changed by less than epsilon
  template <class T> bool operator()(Eigen::MatrixBase<T> const &input) {
    return operator()(input.array());
  }
  //! True if object has changed by less than epsilon
  template <class T> bool operator()(Eigen::ArrayBase<T> const &input) {
    if(previous.size() != input.size()) {
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
  RelativeVariation &epsilon(Real &e) const {
    epsilon_ = e;
    return *this;
  }

protected:
  Real epsilon_;
  Array<Scalar> previous;
};
} /* sopt  */

#endif
