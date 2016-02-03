#ifndef SOPT_SAMPLING_H
#define SOPT_SAMPLING_H

#include "sopt/config.h"
#include "sopt/config.h"
#include <initializer_list>
#include <memory>
#include <random>
#include <Eigen/Core>
#include "sopt/linear_transform.h"
#include "sopt/types.h"

namespace sopt {

//! \brief An operator that samples a set of measurements.
//! \details Picks some elements from a vector
class Sampling {
public:
  //! Constructs from a vector
  Sampling(t_uint size, std::vector<t_uint> const &indices) : indices(indices), size(size) {}
  //! Constructs from the size and the number of samples to pick
  template <class RNG> Sampling(t_uint size, t_uint samples, RNG &&rng);
  //! Constructs from the size and the number of samples to pick
  Sampling(t_uint size, t_uint samples)
      : Sampling(size, samples, std::mt19937_64(std::random_device()())) {}

  // Performs sampling
  template <class T0, class T1>
  void operator()(Eigen::DenseBase<T0> &out, Eigen::DenseBase<T1> const &x) const;
  // Performs sampling
  template <class T0, class T1>
  void operator()(Eigen::DenseBase<T0> &&out, Eigen::DenseBase<T1> const &x) const {
    operator()(out, x);
  }
  // Performs adjunct of sampling
  template <class T0, class T1>
  void adjoint(Eigen::DenseBase<T0> &out, Eigen::DenseBase<T1> const &x) const;
  // Performs adjunct sampling
  template <class T0, class T1>
  void adjoint(Eigen::DenseBase<T0> &&out, Eigen::DenseBase<T1> const &x) const {
    adjoint(out, x);
  }

  //! Size of the vector returned by the adjoint operation
  t_uint cols() const { return size; }
  //! Number of measurements
  t_uint rows() const { return indices.size(); }

protected:
  //! Set of indices to pick
  std::vector<t_uint> indices;
  //! Original vector size
  t_uint size;
};

template <class T0, class T1>
void Sampling::operator()(Eigen::DenseBase<T0> &out, Eigen::DenseBase<T1> const &x) const {
  out.resize(indices.size());
  for(decltype(indices.size()) i(0); i < indices.size(); ++i) {
    assert(indices[i] < x.size());
    out[i] = x[indices[i]];
  }
}

template <class T0, class T1>
void Sampling::adjoint(Eigen::DenseBase<T0> &out, Eigen::DenseBase<T1> const &x) const {
  assert(x.size() == indices.size());
  out.resize(out.size());
  out.fill(0);
  for(decltype(indices.size()) i(0); i < indices.size(); ++i) {
    assert(indices[i] < out.size());
    out[indices[i]] = x[i];
  }
}

//! Returns linear transform version of this object.
template <class T> LinearTransform<Vector<T>> linear_transform(Sampling const &sampling) {
  return linear_transform<Vector<T>>(
      [sampling](Vector<T> &out, Vector<T> const &x) { sampling(out, x); },
      {{0, 1, static_cast<t_int>(sampling.rows())}},
      [sampling](Vector<T> &out, Vector<T> const &x) { sampling.adjoint(out, x); },
      {{0, 1, static_cast<t_int>(sampling.cols())}});
}

template <class RNG>
Sampling::Sampling(t_uint size, t_uint samples, RNG &&rng) : indices(size), size(size) {
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);
  indices.resize(samples);
}

} /* sopt  */
#endif
