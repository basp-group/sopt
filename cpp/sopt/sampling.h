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
      //! Constructor using list of unsigned integers
      Sampling(std::initializer_list<t_uint> const &indices) : indices(indices) {}
      //! Constructs from a vector
      Sampling(std::vector<t_uint> const &indices) : indices(indices) {}
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

      //! Returns linear transform version of this object.
      template<class T> LinearTransform<Eigen::Matrix<T, Eigen::Dynamic, 1>> as_linear_transform() {
        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> t_Vector;
        Sampling const sampling(*this);
        return linear_transform<t_Vector>(
            [sampling](t_Vector &out, t_Vector const &x) { sampling(out, x); },
            [sampling](t_Vector &out, t_Vector const &x) { sampling.adjoint(out, x); }
        );
      }

    protected:
      //! Set of indices to pick
      std::vector<t_uint> indices;
  };

  template<class T0, class T1>
    void Sampling::operator()(Eigen::DenseBase<T0> &out, Eigen::DenseBase<T1> const &x) const {
      assert(out.size() == indices.size());
      for(decltype(indices.size()) i(0); i < indices.size(); ++i) {
        assert(indices[i] < x.size());
        out[i] = x[indices[i]];
      }
    }

  template<class T0, class T1>
    void Sampling::adjoint(Eigen::DenseBase<T0> &out, Eigen::DenseBase<T1> const &x) const {
      assert(x.size() == indices.size());
      for(decltype(indices.size()) i(0); i < indices.size(); ++i) {
        assert(indices[i] < out.size());
        out[indices[i]] = x[i];
      }
    }

} /* sopt  */
#endif

