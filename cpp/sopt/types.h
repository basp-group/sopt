#ifndef BICO_TRAITS_H
#define BICO_TRAITS_H

#include "sopt/config.h"
#include <complex>
#include <functional>
#include <Eigen/Core>
#include "sopt/real_type.h"

namespace sopt {

//! Root of the type hierarchy for signed integers
typedef int t_int;
//! Root of the type hierarchy for unsigned integers
typedef size_t t_uint;
//! Root of the type hierarchy for real numbers
typedef double t_real;
//! Root of the type hierarchy for (real) complex numbers
typedef std::complex<t_real> t_complex;

//! \brief A vector of a given type
//! \details Operates as mathematical vector.
template <class T = t_real> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
//! \brief A matrix of a given type
//! \details Operates as mathematical matrix.
template <class T = t_real> using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
//! \brief A 1-dimensional list of elements of given type
//! \details Operates coefficient-wise, not matrix-vector-wise
template <class T = t_real> using Array = Eigen::Array<T, Eigen::Dynamic, 1>;
//! \brief A 2-dimensional list of elements of given type
//! \details Operates coefficient-wise, not matrix-vector-wise
template <class T = t_real> using Image = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;

//! Typical function out = A*x
template <class VECTOR = Vector<>>
using OperatorFunction = std::function<void(VECTOR &, VECTOR const &)>;
//! Typical function signature for calls to proximal
template <class SCALAR = t_real>
using ProximalFunction = std::function<void(Vector<SCALAR> &, typename real_type<SCALAR>::type,
                                            Vector<SCALAR> const &)>;
//! Typical function signature for convergence
template <class SCALAR = t_real>
using ConvergenceFunction = std::function<bool(Vector<SCALAR> const &)>;
}
#endif
