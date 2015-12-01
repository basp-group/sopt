#ifndef SOPT_PROXIMAL_H
#define SOPT_PROXIMAL_H

#include <iostream>
#include <type_traits>
#include <Eigen/Core>

#include "sopt/utility.h"
#include "sopt/proximal_expression.h"

namespace sopt {
//! Holds some standard proximals
namespace proximal {

//! Proximal of euclidian norm
struct EuclidianNorm {
  template<class T0>
    void operator()(
        Vector<typename T0::Scalar>& out,
        typename real_type<typename T0::Scalar>::type const &t,
        Eigen::MatrixBase<T0> const &x
    ) const {
      typedef typename T0::Scalar Scalar;
      auto const norm = x.stableNorm();
      if(norm > t)
        out = (Scalar(1) - t/norm) * x;
      else
        out.fill(0);
    }
  //! Lazy version
  template<class T0>
    ProximalExpression<EuclidianNorm, T0>
    operator()(typename T0::Scalar const &t, Eigen::MatrixBase<T0> const &x) const {
      return {*this, t, x};
    }
};

//! Proximal of the euclidian norm
template<class T0>
  auto euclidian_norm(
      typename real_type<typename T0::Scalar>::type const & t,
      Eigen::MatrixBase<T0> const &x
  ) -> decltype(EuclidianNorm(), t, x) { return EuclidianNorm()(t, x); }

//! Proximal of the l1 norm
template<class T0, class T1>
  void l1_norm(
      Eigen::DenseBase<T0>& out,
      typename real_type<typename T0::Scalar>::type gamma,
      Eigen::DenseBase<T1> const &x
  ) {
    out = soft_threshhold(x, gamma);
  }

//! \brief Proximal of the l1 norm
//! \detail This specialization makes it easier to use in algorithms, e.g. within `SDMM::append`.
template<class S>
  void l1_norm(Vector<S>& out, typename real_type<S>::type gamma, Vector<S> const &x) {
    l1_norm<Vector<S>, Vector<S>>(out, gamma, x);
  }


//! \brief Proximal of l1 norm
//! \details For more complex version involving linear transforms and weights, see L1TightFrame and
//! L1 classes. In practice, this is an alias for soft_threshhold.
template<class T>
  auto l1_norm(typename real_type<typename T::Scalar>::type gamma, Eigen::DenseBase<T> const &x)
  -> decltype(soft_threshhold(x, gamma)) {
    return soft_threshhold(x, gamma);
  }

//! Proximal for projection on the positive quadrant
template<class T>
  void positive_quadrant(Vector<T>& out, typename real_type<T>::type, Vector<T> const &x) {
    out = sopt::positive_quadrant(x);
  };

//! Proximal for indicator function of L2 ball
template<class T> class L2Ball {
  public:
    typedef typename real_type<T>::type Real;
    //! Constructs an L2 ball proximal of size epsilon
    L2Ball(Real epsilon) : epsilon_(epsilon) {}
    //! Calls proximal function
    void operator()(Vector<T>& out, typename real_type<T>::type, Vector<T> const &x) const {
      return operator()(out, x);
    }
    //! Calls proximal function
    void operator()(Vector<T>& out, Vector<T> const &x) const {
      auto const norm = x.stableNorm();
      if(norm < epsilon())
        out = x;
      else
        out = x * (epsilon() / norm);
    }
    //! Lazy version
    template<class T0>
      ProximalExpression<L2Ball, T0>
      operator()(Real const &t, Eigen::MatrixBase<T0> const &x) const {
        return {*this, t, x};
      }

    //! Size of the ball
    Real epsilon() const { return epsilon_; }
    //! Size of the ball
    L2Ball epsilon(Real eps) { epsilon_ = eps; return *this; }
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
      typename std::enable_if<std::is_reference<OUTPUT>::value, void>::type operator()(
          OUTPUT out,
          typename real_type<typename T0::Scalar>::type const &t,
          Eigen::MatrixBase<T0> const &x
      ) const {
        func(out, t, x + trans);
        out -= trans;
      }
    //! Computes proximal of translated function
    template<class T0>
      void operator()(
          Vector<typename T0::Scalar>& out,
          typename real_type<typename T0::Scalar>::type const &t,
          Eigen::MatrixBase<T0> const &x
      ) const {
        func(out, t, x + trans);
        out -= trans;
      }
    //! Lazy version
    template<class T0>
      ProximalExpression<Translation<FUNCTION, VECTOR> const&, T0>
      operator()(typename T0::Scalar const &t, Eigen::MatrixBase<T0> const &x) const {
        return {*this, t, x};
      }
  private:
    //! Function to translate
    FUNCTION const func;
    //! Translation
    VECTOR const trans;
};

//! Translates given proximal by given vector
template<class FUNCTION, class VECTOR>
  Translation<FUNCTION, VECTOR> translate(FUNCTION const &func, VECTOR const &translation) {
    return Translation<FUNCTION, VECTOR>(func, translation);
  }

}} /* sopt::proximal */

#endif
