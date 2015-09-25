#ifndef SOPT_PROXIMAL_H
#define SOPT_PROXIMAL_H

#include <iostream>
#include <Eigen/Core>
#include "sopt/utility.h"

namespace sopt {
//! Holds some standard proximals
namespace proximal {

namespace details {
  //! Expression referencing a lazy proximal function call
  template<class FUNCTION, class DERIVED>
    class AppliedProximalFunction
      : public Eigen::ReturnByValue<AppliedProximalFunction<FUNCTION, DERIVED>> {
      public:
        typedef typename DERIVED::PlainObject PlainObject;
        typedef typename DERIVED::Index Index;
        typedef typename real_type<typename DERIVED::Scalar>::type Real;

        AppliedProximalFunction(FUNCTION const &func, Real const &gamma, DERIVED const &x)
              : func(func), gamma(gamma), x(x) {}
        AppliedProximalFunction(AppliedProximalFunction const &c)
            : func(c.func), gamma(c.gamma), x(c.x) {}
        AppliedProximalFunction(AppliedProximalFunction &&c)
            : func(std::move(c.func)), gamma(c.gamma), x(c.x) {}

        template<class DESTINATION> void evalTo(DESTINATION &destination) const {
          destination.resizeLike(x);
          func(destination, gamma, x);
        }

        Index rows() const { return x.rows(); }
        Index cols() const { return x.cols(); }

      private:
        FUNCTION const func;
        Real const gamma;
        DERIVED const &x;
    };

} /* details */

//! Proximal of euclidian norm
struct EuclidianNorm {
  template<class VECTOR, class T0>
    void operator()(
        VECTOR &out,
        typename real_type<typename T0::Scalar>::type const &t,
        Eigen::MatrixBase<T0> const &x
    ) const {
      typedef typename T0::Scalar Scalar;
      auto const norm = x.stableNorm();
      if(norm > t)
        out = (Scalar(1) - t/norm) * x;
      else
        out = x.Zero(x.size());
    }
  //! Lazy version
  template<class T0>
    details::AppliedProximalFunction<EuclidianNorm, Eigen::MatrixBase<T0>>
    operator()(typename T0::Scalar const &t, Eigen::MatrixBase<T0> const &x) const {
      typedef details::AppliedProximalFunction<EuclidianNorm, Eigen::MatrixBase<T0>> t_Lazy;
      return t_Lazy(*this, t, x);
    }
};

//! Proximal of the euclidian norm
template<class T0>
  details::AppliedProximalFunction<
    EuclidianNorm,
    Eigen::MatrixBase<T0>
  > euclidian_norm(
      typename real_type<typename T0::Scalar>::type const & t,
      Eigen::MatrixBase<T0> const &x
  ) {
    typedef details::AppliedProximalFunction<EuclidianNorm, Eigen::MatrixBase<T0>>
      t_Lazy;
    return t_Lazy(EuclidianNorm(), t, x);
  }

//! Proximal of the l1 norm
template<class T>
  void l1_norm(
      Eigen::Matrix<T, Eigen::Dynamic, 1> &out,
      typename real_type<T>::type gamma,
      Eigen::Matrix<T, Eigen::Dynamic, 1> const &x
  ) {
    out = soft_threshhold(x, gamma);
  }

//! Proximal for projection on the positive quadrant
template<class T>
  void positive_quadrant(
      Eigen::Matrix<T, Eigen::Dynamic, 1> &out,
      typename real_type<T>::type,
      Eigen::Matrix<T, Eigen::Dynamic, 1> const &x
  ) {
    out = sopt::positive_quadrant(x);
  };

//! Proximal for indicator function of L2 ball
template<class T> class L2Ball {
  public:
    typedef typename real_type<T>::type Real;
    //! Constructs an L2 ball proximal of size epsilon
    L2Ball(Real epsilon) : epsilon_(epsilon) {}
    //! Calls proximal function
    void operator()(
      Eigen::Matrix<T, Eigen::Dynamic, 1> &out,
      typename real_type<T>::type,
      Eigen::Matrix<T, Eigen::Dynamic, 1> const &x
    ) const { return operator()(out, x); }
    //! Calls proximal function
    void operator()(
      Eigen::Matrix<T, Eigen::Dynamic, 1> &out,
      Eigen::Matrix<T, Eigen::Dynamic, 1> const &x
    ) const {
      auto const norm = x.stableNorm();
      out = x * (norm < epsilon_ ? 1e0: epsilon_ / norm);
    }

    //! Size of the ball
    Real epsilon() const { return epsilon_; }
    //! Size of the ball
    L2Ball epsilon(Real eps) const { epsilon_ = eps; return *this; }
  protected:
    //! Size of the ball
    Real epsilon_;
};

//! Translation over proximal function
template<class FUNCTION, class VECTOR> class Translation {
  public:
    //! Creates proximal of translated function
    template<class T_VECTOR>
      Translation(FUNCTION const &func, T_VECTOR const &trans) : func(func), trans(trans) {}
    //! Computes proximal of translated function
    template<class OUTPUT, class T0>
      void operator()(
          OUTPUT &out,
          typename real_type<typename T0::Scalar>::type const &t,
          Eigen::MatrixBase<T0> const &x
      ) const {
        typedef typename T0::Scalar Scalar;
        func(out, t, x + trans);
        out -= trans;
      }
    //! Lazy version
    template<class T0>
      details::AppliedProximalFunction<Translation<FUNCTION, VECTOR> const&, Eigen::MatrixBase<T0>>
      operator()(typename T0::Scalar const &t, Eigen::MatrixBase<T0> const &x) const {
        typedef details::AppliedProximalFunction<
          Translation<FUNCTION, VECTOR> const&, Eigen::MatrixBase<T0>
        > t_Lazy;
        return t_Lazy(*this, t, x);
      }
  private:
    //! Function to translate
    FUNCTION const func;
    //! Translation
    VECTOR const trans;
};

template<class FUNCTION, class VECTOR>
  Translation<FUNCTION, VECTOR> translate(FUNCTION const &func, VECTOR const &translation) {
    return Translation<FUNCTION, VECTOR>(func, translation);
  }

}} /* sopt::proximal */

namespace Eigen { namespace internal {
  template<class FUNCTION, class VECTOR>
    struct traits<sopt::proximal::details::AppliedProximalFunction<FUNCTION, VECTOR>> {
      typedef typename VECTOR::PlainObject ReturnType;
    };
}}

#endif
