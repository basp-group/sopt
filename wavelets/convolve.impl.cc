#include "traits.h"

// Function inside anonymouns namespace won't appear in library
namespace sopt { namespace wavelets { namespace {
    //! \brief Applies scalar product of the size b
    //! \details In practice, the operation is equivalent to the following
    //! - `a` can be seen as a periodic vector: `a[i] == a[i % a.size()]`
    //! - `a'` is the segment `a` = a[offset:offset+b.size()]`
    //! - the result is the scalar product of `a'` by `b`.
    template<class T0, class T2>
      typename T0::Scalar periodic_scalar_product(
          Eigen::MatrixBase<T0> const &a, Eigen::MatrixBase<T2> const &b, t_int offset) {
        auto const Na = static_cast<t_int>(a.size());
        auto const Nb = static_cast<t_int>(b.size());
        // offset in [0, a.size()[
        offset %= Na;
        if(offset < 0) offset += Na;
        // Simple case, just do it
        if(Na > Nb + offset)
          return (a.segment(offset, Nb).transpose() * b);
        // Wrap around, do it, but carefully
        t_real result(0);
        for(t_int i(0), j(offset); i < Nb; ++i, ++j)
          result += a(j % Na) * b(i);
        return result;
      }

    //! \brief Convolves the signal by the filter
    //! \details The signal is seen as a periodic vector during convolution
    template<class T0, class T1, class T2>
      void convolve(
          Eigen::MatrixBase<T0> &result,
          Eigen::MatrixBase<T1> const &signal,
          Eigen::MatrixBase<T2> const &filter)
    {
      assert(result.size() == signal.size());
      for(t_int i(0); i < t_int(signal.size()); ++i)
        result(i) = periodic_scalar_product(signal, filter.reverse(), i);
    }

    //! Convolve and sims low and high pass of a signal
    template<class T0, class T1, class T2, class T3, class T4>
      void convolve_sum(
          Eigen::MatrixBase<T0> &result,
          Eigen::MatrixBase<T1> const &low_pass_signal,
          Eigen::MatrixBase<T2> const &low_pass,
          Eigen::MatrixBase<T3> const &high_pass_signal,
          Eigen::MatrixBase<T4> const &high_pass)
    {
      assert(result.size() == low_pass_signal.size());
      assert(result.size() == high_pass_signal.size());
      assert(low_pass.size() == high_pass.size());

      auto const loffset = low_pass.size() - 1;
      auto const hoffset = high_pass.size() - 1;
      for(t_int i(0); i < static_cast<t_int>(result.size()); ++i)
        result(i) =
          periodic_scalar_product(low_pass_signal, low_pass.reverse(), i - loffset)
          + periodic_scalar_product(high_pass_signal, high_pass.reverse(), i - hoffset);
    }

  }
}}
