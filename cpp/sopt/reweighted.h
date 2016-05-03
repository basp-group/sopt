#ifndef SOPT_REWEIGHTED_H
#define SOPT_REWEIGHTED_H

#include "sopt/linear_transform.h"
#include "sopt/types.h"

namespace sopt {
namespace algorithm {
template <class ALGORITHM> class Reweighted;

//! Factory function to create an l0-approximation by reweighting an l1 norm
template <class ALGORITHM>
Reweighted<ALGORITHM>
reweighted(ALGORITHM const &algo, typename Reweighted<ALGORITHM>::t_SetWeights set_weights,
           typename Reweighted<ALGORITHM>::t_Reweightee reweightee);

//! \brief L0-approximation algorithm, through reweighting
//! \details This algorithm approximates \f$min_x ||Ψ^Tx||_0 + f(x)\f$ by solving the set of
//! problems \f$j\f$, \f$min_x ||W_jΨ^Tx||_1 + f(x)\f$ where the *diagonal* matrix \f$W_j\f$ is set
//! using the results from \f$j-1\f$: \f$ δ_j W_j^{-1} = δ_j + ||W_{j-1}Ψ^T||_1\f$. \f$δ_j\f$
//! prevents division by zero. It is a series which converges to zero. By default,
//! \f$δ_{j+1}=0.1δ_j\f$.
//!
//! The algorithm proceeds needs three forms of input:
//! - the inner algorithm, e.g. ImagingProximalADMM
//! - a function returning Ψ^Tx given x
//! - a function to modify the inner algorithm with new weights
template <class ALGORITHM> class Reweighted {
public:
  //! Inner-loop algorithm
  typedef ALGORITHM Algorithm;
  //! Scalar type
  typedef typename Algorithm::Scalar Scalar;
  //! Real type
  typedef typename real_type<Scalar>::type Real;
  //! Type of then underlying vectors
  typedef typename Algorithm::t_Vector t_Vector;
  //! Type of the convergence function
  typedef typename Algorithm::t_IsConverged t_IsConverged;
  //! \brief Type of the function that is subject to reweighting
  //! \details E.g. \f$Ψ^Tx\f$. Note that l1-norm is not applied here.
  typedef std::function<t_Vector(Algorithm const &, t_Vector const &)> t_Reweightee;
  //! Type of the function to set weights
  typedef std::function<void(Algorithm &, t_Vector const &)> t_SetWeights;
  //! Function to update delta at each turn
  typedef std::function<Real(Real)> t_DeltaUpdate;

  //! output from running reweighting scheme
  struct ReweightedResult {
    //! Number of iterations (outer loop)
    t_uint niters;
    //! Wether convergence was achieved
    bool good;
    //! Weights at last iteration
    t_Vector weights;
    //! Result from last inner loop
    typename Algorithm::DiagnosticAndResult algo;
    //! Default construction
    ReweightedResult() : niters(0), good(false), weights(t_Vector::Ones(1)), algo() {}
  };

  Reweighted(Algorithm const &algo, t_SetWeights const &setweights, t_Reweightee const &reweightee)
      : algo_(algo), setweights_(setweights), reweightee_(reweightee),
        itermax_(std::numeric_limits<t_uint>::max()), noise_(0e0), is_converged_(),
        update_delta_([](Real delta) { return 1e-1 * delta; }) {}

  //! Underlying "inner-loop" algorithm
  Algorithm &algorithm() { return algo_; }
  //! Underlying "inner-loop" algorithm
  Algorithm const &algorithm() const { return algo_; }
  //! Sets the underlying "inner-loop" algorithm
  Reweighted<Algorithm> &algorithm(Algorithm const &algo) {
    algo_ = algo;
    return *this;
  }
  //! Sets the underlying "inner-loop" algorithm
  Reweighted<Algorithm> &algorithm(Algorithm &&algo) {
    algo_ = std::move(algo);
    return *this;
  }

  //! Function to reset the weights in the algorithm
  t_SetWeights const &set_weights() const { return setweights_; }
  //! Function to reset the weights in the algorithm
  Reweighted<Algorithm> &set_weights(t_SetWeights const &setweights) const {
    setweights_ = setweights;
    return *this;
  }
  //! Sets the weights on the underlying algorithm
  void set_weights(Algorithm &algo, t_Vector const &weights) const {
    return set_weights()(algo, weights);
  }

  //! Function that needs to be reweighted
  //! \details E.g. \f$Ψ^Tx\f$. Note that l1-norm is not applied here.
  Reweighted<Algorithm> &reweightee(t_Reweightee const &rw) {
    reweightee_ = rw;
    return *this;
  }
  //! Function that needs to be reweighted
  t_Reweightee const &reweightee() const { return reweightee_; }
  //! Forwards to the reweightee function
  t_Vector reweightee(t_Vector const &x) const { return reweightee()(algorithm(), x); }

  //! Maximum number of reweighted iterations
  t_uint itermax() const { return itermax_; }
  Reweighted &itermax(t_uint i) {
    itermax_ = i;
    return *this;
  }
  //! \brief weights are set to noise / (noise + x), more or less
  Real noise() const { return noise_; }
  Reweighted &noise(Real noise) {
    noise_ = noise;
    return *this;
  }
  //! Checks convergence of the reweighting scheme
  t_IsConverged const &is_converged() const { return is_converged_; }
  Reweighted &is_converged(t_IsConverged const &convergence) {
    is_converged_ = convergence;
    return *this;
  }
  bool is_converged(t_Vector const &x) const { return is_converged() ? is_converged()(x) : false; }

  //! \brief Performs reweighting
  //! \details This overload will compute an initial result without initial weights set to one.
  template <class INPUT>
  typename std::enable_if<not(std::is_same<INPUT, typename Algorithm::DiagnosticAndResult>::value
                              or std::is_same<INPUT, ReweightedResult>::value),
                          ReweightedResult>::type
  operator()(INPUT const &input) const;
  //! \brief Performs reweighting
  //! \details This overload will compute an initial result without initial weights set to one.
  ReweightedResult operator()() const;
  //! Reweighted algorithm, from prior call to inner-algorithm
  ReweightedResult operator()(typename Algorithm::DiagnosticAndResult const &warm) const;
  //! Reweighted algorithm, from prior call to reweighting algorithm
  ReweightedResult operator()(ReweightedResult const &warm) const;

  //! Updates delta
  Real update_delta(Real delta) const { return update_delta()(delta); }
  //! Updates delta
  t_DeltaUpdate const &update_delta() const { return update_delta_; }
  //! Updates delta
  Reweighted<Algorithm> update_delta(t_DeltaUpdate const &ud) const { return update_delta_ = ud; }

protected:
  //! Inner loop algorithm
  Algorithm algo_;
  //! Function to set weights
  t_SetWeights setweights_;
  //! \brief Function that is subject to reweighting
  //! \details E.g. \f$Ψ^Tx\f$. Note that l1-norm is not applied here.
  t_Reweightee reweightee_;
  //! Maximum number of reweighted iterations
  t_uint itermax_;
  //! \brief weights are set to noise / (noise + x)
  //! \details If noise is <= 0e0, then it is computed in current_noise
  Real noise_;
  //! Checks convergence
  t_IsConverged is_converged_;
  //! Updates delta at each turn
  t_DeltaUpdate update_delta_;
};

template <class ALGORITHM>
template <class INPUT>
typename std::
    enable_if<not(std::is_same<INPUT, typename ALGORITHM::DiagnosticAndResult>::value
                  or std::is_same<INPUT, typename Reweighted<ALGORITHM>::ReweightedResult>::value),
              typename Reweighted<ALGORITHM>::ReweightedResult>::type
    Reweighted<ALGORITHM>::operator()(INPUT const &input) const {
  Algorithm algo = algorithm();
  set_weights(algo, t_Vector::Ones(1));
  return operator()(algo(input));
}

template <class ALGORITHM>
typename Reweighted<ALGORITHM>::ReweightedResult Reweighted<ALGORITHM>::operator()() const {
  Algorithm algo = algorithm();
  set_weights(algo, t_Vector::Ones(1));
  return operator()(algo());
}

template <class ALGORITHM>
typename Reweighted<ALGORITHM>::ReweightedResult Reweighted<ALGORITHM>::
operator()(typename Algorithm::DiagnosticAndResult const &warm) const {
  ReweightedResult result;
  result.algo = warm;
  result.weights = t_Vector::Ones(1);
  return operator()(result);
}

template <class ALGORITHM>
typename Reweighted<ALGORITHM>::ReweightedResult Reweighted<ALGORITHM>::
operator()(ReweightedResult const &warm) const {
  SOPT_NOTICE("Starting reweighted scheme");
  // Copies inner algorithm, so that operator() can be constant
  Algorithm algo(algorithm());
  ReweightedResult result(warm);

  auto delta = std::max(standard_deviation(reweightee(warm.algo.x)), noise());
  SOPT_WARN("-   Initial delta: {}", delta);
  for(result.niters = 0; result.niters < itermax(); ++result.niters) {
    SOPT_INFO("Reweigting iteration {}/{} ", result.niters, itermax());
    SOPT_WARN("  - delta: {}", delta);
    result.weights = delta / (delta + reweightee(result.algo.x).array().abs());
    set_weights(algo, result.weights);
    result.algo = algo(result.algo);
    if(is_converged(result.algo.x)) {
      SOPT_INFO("Reweighting scheme did converge in {} iterations", result.niters);
      result.good = true;
      break;
    }
    delta = update_delta(delta);
  }
  // result is always good if no convergence function is defined
  if(not is_converged())
    result.good = true;
  else if(not result.good)
    SOPT_INFO("Reweighting scheme did *not* converge in {} iterations", itermax());
  return result;
}

//! Factory function to create an l0-approximation by reweighting an l1 norm
template <class ALGORITHM>
Reweighted<ALGORITHM>
reweighted(ALGORITHM const &algo, typename Reweighted<ALGORITHM>::t_SetWeights set_weights,
           typename Reweighted<ALGORITHM>::t_Reweightee reweightee) {
  return {algo, set_weights, reweightee};
}

template <class SCALAR> class ImagingProximalADMM;

//! Specialized version for reweighting the imaging PADMM
template <class SCALAR>
Reweighted<ImagingProximalADMM<SCALAR>> reweighted(ImagingProximalADMM<SCALAR> const &algo) {

  auto const reweightee = [](ImagingProximalADMM<SCALAR> const &algo,
                             typename ImagingProximalADMM<SCALAR>::t_Vector const &x) {
    return (algo.Phi().adjoint() * x).eval();
  };
  auto const set_weights = [](ImagingProximalADMM<SCALAR> &algo,
                              typename ImagingProximalADMM<SCALAR>::t_Vector const &weights) {
    algo.l1_proximal_weights(weights);
  };
  return {algo, set_weights, reweightee};
}
} // namespace algorithm
} // namespace sopt
#endif
