#include "traits.h"

// Function inside anonymouns namespace won't appear in library
namespace sopt { namespace wavelets { namespace {
    //! Applies scalar product of the size b
    //! Elements from a are wrapped around to match b
    template<class T0, class T2>
      typename T0::Scalar periodic_scalar_product(
          Eigen::MatrixBase<T0> const &a, Eigen::MatrixBase<T2> const &b, t_uint offset) {
        // Simple, just do it
        if(a.size() > b.size() + offset)
          return (a.segment(offset, b.size()).transpose() * b);
        t_real result(0);
        for(t_uint i(0); i < t_uint(b.size()); ++i)
          result += a((i+offset) % a.size()) * b(i);
        return result;
      }

    // Convolve low or high pass
    template<class T0, class T1, class T2>
      void convolve(
          Eigen::MatrixBase<T0> &result,
          Eigen::MatrixBase<T1> const &signal,
          Eigen::MatrixBase<T2> const &filter)
    {
      assert(result.size() == signal.size());
      for(t_uint i(0); i < t_uint(signal.size()); ++i)
        result(i) = periodic_scalar_product(signal, filter.reverse(), i);
    }
  }
}}
