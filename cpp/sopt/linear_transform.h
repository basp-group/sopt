#ifndef SOPT_OPERATORS_H
#define SOPT_OPERATORS_H

#include <type_traits>
#include <array>
#include <memory>

#include <Eigen/Core>
#include "sopt/types.h"
#include "sopt/wrapper.h"
#include "sopt/utility.h"
#include "sopt/logging.h"

namespace sopt {

namespace details {
  //! \brief Wraps a matrix into a function and its conjugate transpose
  //! \details This class helps to wrap matrices into functions, such that we can use and store them
  //! such that SDMM algorithms can refer to them.
  template<class EIGEN> class MatrixToLinearTransform;
  //! Wraps a tranposed matrix into a function and its conjugate transpose
  template<class EIGEN> class MatrixAdjointToLinearTransform;
}

//! Joins together direct and indirect operators
template<class VECTOR> class LinearTransform : public details::WrapFunction<VECTOR> {
  public:
    //! Type of the wrapped functions
    typedef std::function<void(VECTOR &, VECTOR const &)> t_Function;

    //! Constructor
    //! \param[in] direct: function with signature void(VECTOR&, VECTOR const&) which applies a
    //!    linear operator to a vector.
    //! \param[in] indirect: function with signature void(VECTOR&, VECTOR const&) which applies a
    //!    the conjugate transpose linear operator to a vector.
    //! \param[in] sizes: 3 integer elements (a, b, c) such that if the input to linear operator is
    //!     of size N, then the output is of size (a * N) / b + c. A similar quantity is deduced for
    //!     the indirect operator.
    LinearTransform(
        t_Function const &direct, t_Function const &indirect,
        std::array<t_int, 3> sizes = {{1, 1, 0}}
    ) : LinearTransform(
          direct, sizes,
          indirect, {{sizes[1], sizes[0], sizes[0] == 0? 0: -(sizes[2]*sizes[1])/sizes[0]}}
        ) {
      assert(sizes[0] != 0);
    }
    //! Constructor
    //! \param[in] direct: function with signature void(VECTOR&, VECTOR const&) which applies a
    //!    linear operator to a vector.
    //! \param[in] dsizes: 3 integer elements (a, b, c) such that if the input to the linear
    //!    operator is of size N, then the output is of size (a * N) / b + c.
    //! \param[in] indirect: function with signature void(VECTOR&, VECTOR const&) which applies a
    //!    the conjugate transpose linear operator to a vector.
    //! \param[in] dsizes: 3 integer elements (a, b, c) such that if the input to the indirect
    //!    linear operator is of size N, then the output is of size (a * N) / b + c.
    LinearTransform(
        t_Function const &direct, std::array<t_int, 3> dsizes,
        t_Function const &indirect, std::array<t_int, 3> isizes
    ) : LinearTransform(details::wrap(direct, dsizes), details::wrap(indirect, isizes)) {}
    LinearTransform(
        details::WrapFunction<VECTOR> const &direct,
        details::WrapFunction<VECTOR> const &indirect
    ) : details::WrapFunction<VECTOR>(direct), indirect_(indirect) {}
    LinearTransform(LinearTransform const &c)
      : details::WrapFunction<VECTOR>(c), indirect_(c.indirect_) {}
    LinearTransform(LinearTransform &&c)
      : details::WrapFunction<VECTOR>(std::move(c)), indirect_(std::move(c.indirect_)) {}
    void operator=(LinearTransform const &c) {
      details::WrapFunction<VECTOR>::operator=(c);
      indirect_ = c.indirect_;
    }
    void operator=(LinearTransform &&c) {
      details::WrapFunction<VECTOR>::operator=(std::move(c));
      indirect_ = std::move(c.indirect_);
    }

    //! Indirect transform
    details::WrapFunction<VECTOR> adjoint() const { return indirect_; }

    using details::WrapFunction<VECTOR>::operator*;
    using details::WrapFunction<VECTOR>::sizes;
    using details::WrapFunction<VECTOR>::rows;

  private:
    //! Function applying conjugate transpose operator
    details::WrapFunction<VECTOR> indirect_;
};

//! Helper function to creates a function operator
//! \param[in] direct: function with signature void(VECTOR&, VECTOR const&) which applies a
//!    linear operator to a vector.
//! \param[in] indirect: function with signature void(VECTOR&, VECTOR const&) which applies a
//!    the conjugate transpose linear operator to a vector.
//! \param[in] sizes: 3 integer elements (a, b, c) such that if the input to linear operator is
//!     of size N, then the output is of size (a * N) / b + c. A similar quantity is deduced for
//!     the indirect operator.
template<class VECTOR>
  LinearTransform<VECTOR> linear_transform(
      std::function<void(VECTOR&, VECTOR const&)> const& direct,
      std::function<void(VECTOR&, VECTOR const&)> const& indirect,
      std::array<t_int, 3> const &sizes = {{1, 1, 0}}
  ) { return {direct, indirect, sizes}; }
//! Helper function to creates a function operator
 //! \param[in] direct: function with signature void(VECTOR&, VECTOR const&) which applies a
 //!    linear operator to a vector.
 //! \param[in] dsizes: 3 integer elements (a, b, c) such that if the input to the linear
 //!    operator is of size N, then the output is of size (a * N) / b + c.
 //! \param[in] indirect: function with signature void(VECTOR&, VECTOR const&) which applies a
 //!    the conjugate transpose linear operator to a vector.
 //! \param[in] dsizes: 3 integer elements (a, b, c) such that if the input to the indirect
 //!    linear operator is of size N, then the output is of size (a * N) / b + c.
template<class VECTOR>
  LinearTransform<VECTOR> linear_transform(
      std::function<void(VECTOR&, VECTOR const&)> const& direct,
      std::array<t_int, 3> const &dsizes,
      std::function<void(VECTOR&, VECTOR const&)> const& indirect,
      std::array<t_int, 3> const &isizes
  ) { return {direct, dsizes, indirect, isizes}; }

//! Convenience no-op function
template<class VECTOR>
  LinearTransform<VECTOR>& linear_transform(LinearTransform<VECTOR> &passthrough) {
    return passthrough;
  }
//! Creates a linear transform from a pair of wrappers
template<class VECTOR>
  LinearTransform<VECTOR> linear_transform(
      details::WrapFunction<VECTOR> const &direct, details::WrapFunction<VECTOR> const &adjoint ) {
    return {direct, adjoint};
  }


namespace details {

  template<class EIGEN> class MatrixToLinearTransform {
      //! The underlying raw matrix type
      typedef typename std::remove_const<typename std::remove_reference<EIGEN>::type>::type Raw;
      //! The matrix underlying the expression
      typedef typename Raw::PlainObject PlainMatrix;
    public:
      //! The output type
      typedef typename std::conditional<
        std::is_base_of<Eigen::MatrixBase<PlainMatrix>, PlainMatrix>::value,
        Vector<typename PlainMatrix::Scalar>, Array<typename PlainMatrix::Scalar>
      >::type PlainObject;
      //! \brief Creates from an expression
      //! \details Expression is evaluated and the result stored internally. This object owns a
      //! copy of the matrix. It might share it with a few friendly neighbors.
      template<class T0>
        MatrixToLinearTransform(Eigen::MatrixBase<T0> const& A)
          : matrix(std::make_shared<EIGEN>(A)) {}
      //! Creates from a shared matrix.
      MatrixToLinearTransform(std::shared_ptr<EIGEN> const &x) : matrix(x) {};

      //! Performs operation
      void operator()(PlainObject &out, PlainObject const& x) const {
        out = (*matrix) * x;
      }
      //! \brief Returns conjugate transpose operator
      //! \details The matrix is shared.
      MatrixAdjointToLinearTransform<EIGEN> adjoint() const {
        return MatrixAdjointToLinearTransform<EIGEN>(matrix);
      }
    private:
      //! Wrapped matrix
      std::shared_ptr<EIGEN> matrix;
  };

  template<class EIGEN> class MatrixAdjointToLinearTransform {
    public:
      typedef typename MatrixToLinearTransform<EIGEN>::PlainObject PlainObject;
      //! \brief Creates from an expression
      //! \details Expression is evaluated and the result stored internally. This object owns a
      //! copy of the matrix. It might share it with a few friendly neighbors.
      template<class T0>
        MatrixAdjointToLinearTransform(Eigen::MatrixBase<T0> const& A)
          : matrix(std::make_shared<EIGEN>(A)) {}
      //! Creates from a shared matrix.
      MatrixAdjointToLinearTransform(std::shared_ptr<EIGEN> const &x) : matrix(x) {};

      //! Performs operation
      void operator()(PlainObject &out, PlainObject const& x) const {
        out = matrix->transpose().conjugate() * x;
      }
      //! \brief Returns adjoint operator
      //! \details The matrix is shared.
      MatrixToLinearTransform<EIGEN> adjoint() const {
        return MatrixToLinearTransform<EIGEN>(matrix);
      }

    private:
      std::shared_ptr<EIGEN> matrix;
  };
}

//! Helper function to creates a function operator
template<class DERIVED>
  LinearTransform<Vector<typename DERIVED::Scalar>>
  linear_transform(Eigen::MatrixBase<DERIVED> const &A) {
    details::MatrixToLinearTransform<Matrix<typename DERIVED::Scalar>> const matrix(A);
    if(A.rows() == A.cols())
      return {matrix, matrix.adjoint()};
    else {
      t_int const gcd = details::gcd(A.cols(), A.rows());
      t_int const a = A.cols() / gcd;
      t_int const b = A.rows() / gcd;
      return {matrix, matrix.adjoint(), {{b, a, 0}}};
    }
  }
}
#endif
