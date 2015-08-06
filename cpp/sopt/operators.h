#ifndef SOPT_OPERATORS_H
#define SOPT_OPERATORS_H

#include <type_traits>
#include <Eigen/Core>
#include "types.h"
#include "wrapper.h"

namespace sopt {

namespace details {
  //! \brief Wraps a matrix into a function and its transpose
  //! \details This class helps to wrap matrices into functions, such that we can use and store them
  //! such that SDMM algorithms can refer t 
  template<class EIGEN> class MatrixToOperator;
  //! Wraps a tranposed matrix into a function and its transpose
  template<class EIGEN> class MatrixTransposeToOperator;
}

//! Joins together direct and indirect operators
template<class VECTOR> class Operator : public details::WrapFunction<VECTOR> {
  public:
    //! Type of the wrapped functions
    typedef std::function<void(VECTOR &, VECTOR const &)> t_Function;

    Operator(t_Function const &direct, t_Function const &transpose)
      : details::WrapFunction<VECTOR>(direct), transpose_(transpose) {}

    //! Indirect transform
    details::WrapFunction<VECTOR> transpose() const { return details::wrap(transpose_); }
    using details::WrapFunction<VECTOR>::operator*;

  private:
    //! Function applying transpose operator
    t_Function const transpose_;
};

//! Joins together direct, indirect, and proximal operators
template<class VECTOR> class ProximalOperator : public Operator<VECTOR> {
  public:
    //! Type of the wrapped functions
    typedef typename Operator<VECTOR>::t_Function t_Function;

    ProximalOperator(t_Function const &dir, t_Function const &trans, t_Function const &prox)
      : Operator<VECTOR>(dir, trans), proximal_(prox) {}

    //! Proximal transform
    template<class T0>
      details::AppliedFunction<t_Function const&, Eigen::ArrayBase<T0>> proximal(
          Eigen::ArrayBase<T0> const &x) const {
        return details::AppliedFunction<t_Function const&, Eigen::ArrayBase<T0>>(proximal_, x);
      }
    //! Proximal transform
    template<class T0>
      details::AppliedFunction<t_Function const&, Eigen::MatrixBase<T0>> proximal(
          Eigen::MatrixBase<T0> const &x) const {
        return details::AppliedFunction<t_Function const&, Eigen::MatrixBase<T0>>(proximal_, x);
      }

    using Operator<VECTOR>::operator*;
    using Operator<VECTOR>::transpose;

  private:
    //! Function applying proximal operator
    t_Function const proximal_;
};

//! Helper function to creates a function operator
template<class VECTOR>
  Operator<VECTOR> make_operator(
      typename ProximalOperator<VECTOR>::t_Function const& direct,
      typename ProximalOperator<VECTOR>::t_Function const& indirect
  ) { return Operator<VECTOR>(direct, indirect); }


//! Helper function to creates a function operator
template<class VECTOR>
  ProximalOperator<VECTOR> make_operator(
      typename ProximalOperator<VECTOR>::t_Function const& direct,
      typename ProximalOperator<VECTOR>::t_Function const& indirect,
      typename ProximalOperator<VECTOR>::t_Function const& proximal
  ) { return ProximalOperator<VECTOR>(direct, indirect, proximal); }

namespace details {

  template<class EIGEN> class MatrixToOperator {
      //! The underlying raw matrix type
      typedef typename std::remove_const<typename std::remove_reference<EIGEN>::type>::type Raw;
      //! The matrix underlying the expression
      typedef typename Raw::PlainObject PlainMatrix;
    public:
      //! The output type
      typedef typename std::conditional<
        std::is_base_of<Eigen::MatrixBase<PlainMatrix>, PlainMatrix>::value,
        Eigen::Matrix<typename PlainMatrix::Scalar, Eigen::Dynamic, 1>,
        Eigen::Array<typename PlainMatrix::Scalar, Eigen::Dynamic, 1>
      >::type PlainObject;
      //! \brief Creates from an expression
      //! \details Expression is evaluated and the result stored internally. This object owns a
      //! copy of the matrix. It might share it with a few friendly neighbors.
      template<class T0>
        MatrixToOperator(Eigen::MatrixBase<T0> const& A) : matrix(std::make_shared<EIGEN>(A)) {}
      //! Creates from a shared matrix.
      MatrixToOperator(std::shared_ptr<EIGEN> const &x) : matrix(x) {};

      //! Performs operation
      void operator()(PlainObject &out, PlainObject const& x) const {
        out = (*matrix) * x;
      }
      //! \brief Returns transpose operator
      //! \details The matrix is shared.
      MatrixTransposeToOperator<EIGEN> transpose() const {
        return MatrixTransposeToOperator<EIGEN>(matrix);
      }
    private:
      //! Wrapped matrix
      std::shared_ptr<EIGEN> matrix;
  };

  template<class EIGEN> class MatrixTransposeToOperator {
    public:
      typedef typename MatrixToOperator<EIGEN>::PlainObject PlainObject;
      //! \brief Creates from an expression
      //! \details Expression is evaluated and the result stored internally. This object owns a
      //! copy of the matrix. It might share it with a few friendly neighbors.
      template<class T0>
        MatrixTransposeToOperator(Eigen::MatrixBase<T0> const& A) : matrix(std::make_shared<EIGEN>(A)) {}
      //! Creates from a shared matrix.
      MatrixTransposeToOperator(std::shared_ptr<EIGEN> const &x) : matrix(x) {};

      //! Performs operation
      void operator()(PlainObject &out, PlainObject const& x) const {
        out = matrix->transpose() * x;
      }
      //! \brief Returns transpose operator
      //! \details The matrix is shared.
      MatrixToOperator<EIGEN> transpose() const { return MatrixToOperator<EIGEN>(matrix); }

    private:
      std::shared_ptr<EIGEN> matrix;
  };
}

//! Helper function to creates a function operator
template<class DERIVED>
  auto make_operator(Eigen::MatrixBase<DERIVED> const &A)
  -> Operator<Eigen::Matrix<typename DERIVED::Scalar, Eigen::Dynamic, 1>> {
    typedef Eigen::Matrix<typename DERIVED::Scalar, Eigen::Dynamic, 1> t_Vector;
    typedef Eigen::Matrix<typename DERIVED::Scalar, Eigen::Dynamic, Eigen::Dynamic> t_Matrix;
    details::MatrixToOperator<t_Matrix> const matrix(A);
    return Operator<t_Vector>(matrix, matrix.transpose());
  }
}
#endif
