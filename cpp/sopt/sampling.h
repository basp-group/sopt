#ifndef SOPT_SAMPLING_H
#define SOPT_SAMPLING_H

#include <random>
#include <initializer_list>

#include <Eigen/Core>

#include "sopt/types.h"
#include "sopt/linear_transform.h"

namespace sopt {

//! \brief An operator that samples a set of measurements.
//! \details Picks some elements from a vector
class Sampling {
  public:
    //! Constructs from a vector
    Sampling(t_uint size, std::vector<t_uint> const &indices) : indices(indices), size(size) {}
    //! Constructs from the size and the number of samples to pick
    Sampling(t_uint size, t_uint samples);

    // Performs sampling
    template<class T0, class T1>
      void operator()(Eigen::DenseBase<T0> &out, Eigen::DenseBase<T1> const &x) const;
    // Performs sampling
    template<class T0, class T1>
      void operator()(Eigen::DenseBase<T0> &&out, Eigen::DenseBase<T1> const &x) const {
        operator()(out, x);
      }
    // Performs adjunct of sampling
    template<class T0, class T1>
      void adjoint(Eigen::DenseBase<T0> &out, Eigen::DenseBase<T1> const &x) const;
    // Performs adjunct sampling
    template<class T0, class T1>
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

template<class T0, class T1>
  void Sampling::operator()(Eigen::DenseBase<T0> &out, Eigen::DenseBase<T1> const &x) const {
    out.resize(indices.size());
    for(decltype(indices.size()) i(0); i < indices.size(); ++i) {
      assert(indices[i] < x.size());
      out[i] = x[indices[i]];
    }
  }

template<class T0, class T1>
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
template<class T>
  LinearTransform<Eigen::Matrix<T, Eigen::Dynamic, 1>> linear_transform(Sampling const &sampling) {
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> t_Vector;
    return linear_transform<t_Vector>(
        [sampling](t_Vector &out, t_Vector const &x) { sampling(out, x); },
        {{0, 1, static_cast<t_int>(sampling.rows())}},
        [sampling](t_Vector &out, t_Vector const &x) { sampling.adjoint(out, x); },
        {{0, 1, static_cast<t_int>(sampling.cols())}}
    );
  }
} /* sopt  */
#endif

