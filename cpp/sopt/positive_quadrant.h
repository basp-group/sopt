#ifndef SOPT_PROJECTED_ALGORITHM_H
#define SOPT_PROJECTED_ALGORITHM_H

#include "sopt/linear_transform.h"
#include "sopt/types.h"

namespace sopt {
namespace algorithm {
//! \brief Computes according to given algorithm and then projects it to the positive quadrant
//! \details C implementation of the reweighted algorithms uses this, even-though the solutions are
//! already constrained to the positive quadrant.
template <class ALGORITHM> class PositiveQuadrant {
public:
  //! Underlying algorithm
  typedef ALGORITHM Algorithm;
  //! Underlying scalar
  typedef typename Algorithm::Scalar Scalar;
  //! Underlying vector
  typedef typename Algorithm::t_Vector t_Vector;
  //! Underlying convergence functions
  typedef typename Algorithm::t_IsConverged t_IsConverged;
  //! Underlying result type
  typedef typename ALGORITHM::Diagnostic Diagnostic;
  //! Underlying result type
  typedef typename ALGORITHM::DiagnosticAndResult DiagnosticAndResult;

  PositiveQuadrant(Algorithm const &algo) : algorithm_(algo) {}
  PositiveQuadrant(Algorithm &&algo) : algorithm_(std::move(algo)) {}

  Algorithm &algorithm() { return algorithm_; }
  Algorithm const &algorithm() const { return algorithm_; }

  //! Performs algorithm and project results onto positive quadrant
  template <class... T> Diagnostic operator()(t_Vector &out, T const &... args) const {
    auto const diagnostic = algorithm()(out, std::forward<T const &>(args)...);
    out = positive_quadrant(out);
    return diagnostic;
  };

  //! Performs algorithm and project results onto positive quadrant
  template <class... T> DiagnosticAndResult operator()(T const &... args) const {
    auto result = algorithm()(std::forward<T const &>(args)...);
    result.x = positive_quadrant(result.x);
    return result;
  };

protected:
  //! Underlying algorithm
  Algorithm algorithm_;
};

//! Extended algorithm where the solution is projected on the positive quadrant
template <class ALGORITHM> PositiveQuadrant<ALGORITHM> positive_quadrant(ALGORITHM const &algo) {
  return {algo};
}
}
}

#endif
