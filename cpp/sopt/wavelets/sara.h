#ifndef SOPT_WAVELETS_SARA_H
#define SOPT_WAVELETS_SARA_H

#include "sopt/config.h"
#include <cmath>
#include <initializer_list>
#include <tuple>
#include <vector>
#include "sopt/logging.h"
#include "sopt/wavelets/wavelets.h"

namespace sopt {
namespace wavelets {

//! Sparsity Averaging Reweighted Analysis
class SARA : public std::vector<Wavelet> {
public:
#ifndef SOPT_HAS_NOT_USING
  // Constructors
  using std::vector<Wavelet>::vector;
#else
  //! Default constructor
  SARA() : std::vector<Wavelet>(){};
#endif
  //! Easy constructor
  SARA(std::initializer_list<std::tuple<std::string, t_uint>> const &init)
      : SARA(init.begin(), init.end()) {}
  //! Construct from any iterator over a (std:string, t_uint) tuple
  template <class ITERATOR,
            class T = typename std::
                enable_if<std::is_convertible<decltype(std::get<0>(*std::declval<ITERATOR>())),
                                              std::string>::value
                          and std::is_convertible<decltype(std::get<1>(*std::declval<ITERATOR>())),
                                                  t_uint>::value>::type>
  SARA(ITERATOR first, ITERATOR last) {
    for(; first != last; ++first)
      emplace_back(std::get<0>(*first), std::get<1>(*first));
  }
  //! Destructor
  virtual ~SARA() {}

  //! \brief Direct transform
  //! \param[in] signal: computes wavelet coefficients for this signal. Its size must be a
  //! multiple of $2^l$ where $l$ is the maximum number of levels. Can be a matrix (2d-transform)
  //! or a column vector (1-d transform).
  //! \return wavelets coefficients arranged by columns: if the input is n by m, then the output
  //! is n by m * d, with d the number of wavelets.
  //! \details Supports 1 and 2 dimensional tranforms for real and complex data.
  template <class T0> typename T0::PlainObject direct(Eigen::ArrayBase<T0> const &signal) const;
  //! \brief Direct transform
  //! \param[inout] coefficients: Output wavelet coefficients. Must be of the type as the input.
  //! If the input is n by m, and d is the number of wavelets, then the output should be n by (m *
  //! d).
  //! \param[in] signal: computes wavelet coefficients for this signal. Its size must be a
  //! multiple of $2^l$ where $l$ is the maximum number of levels. Can be a matrix (2d-transform)
  //! or a column vector (1-d transform).
  //! \details Supports 1 and 2 dimensional tranforms for real and complex data.
  template <class T0, class T1>
  void direct(Eigen::ArrayBase<T1> &coefficients, Eigen::ArrayBase<T0> const &signal) const;
  //! \brief Direct transform
  //! \param[inout] coefficients: Output wavelet coefficients. Must be of the type as the input.
  //! If the input is n by m, and l is the number of wavelets, then the output should be n by (m *
  //! l).
  //! \param[in] signal: computes wavelet coefficients for this signal. Its size must be a
  //! multiple of $2^l$ where $l$ is the number of levels. Can be a matrix (2d-transform) or a
  //! column vector (1-d transform).
  //! \details Supports 1 and 2 dimensional tranforms for real and complex data. This version
  //! allows non-constant Eigen expressions to be passe on without the ugly `const_cast` of the
  //! cannonical approach.
  template <class T0, class T1>
  void direct(Eigen::ArrayBase<T1> &&coefficients, Eigen::ArrayBase<T0> const &signal) const {
    direct(coefficients, signal);
  }
  //! \brief Indirect transform
  //! \param[in] coefficients: Input wavelet coefficients. Its size must be a multiple of $2^l$
  //! where $l$ is the number of levels. Can be a matrix (2d-transform) or a column vector (1-d
  //! transform).
  //! \details Supports 1 and 2 dimensional tranforms for real and complex data.
  template <class T0> typename T0::PlainObject indirect(Eigen::ArrayBase<T0> const &coeffs) const;
  //! \brief Indirect transform
  //! \param[in] coefficients: Input wavelet coefficients. Its size must be a multiple of $2^l$
  //! where $l$ is the number of levels. Can be a matrix (2d-transform) or a column vector (1-d
  //! \param[inout] signal: Reconstructed signal. Must be of the same size and type as the input.
  //! \details Supports 1 and 2 dimensional tranforms for real and complex data.
  template <class T0, class T1>
  void indirect(Eigen::ArrayBase<T1> const &coefficients, Eigen::ArrayBase<T0> &signal) const;
  //! \brief Indirect transform
  //! \param[in] coefficients: Input wavelet coefficients. Its size must be a multiple of $2^l$
  //! where $l$ is the number of levels. Can be a matrix (2d-transform) or a column vector (1-d
  //! \param[inout] signal: Reconstructed signal. Must be of the same size and type as the input.
  //! \details Supports 1 and 2 dimensional tranforms for real and complex data.  This version
  //! allows non-constant Eigen expressions to be passe on without the ugly `const_cast` of the
  //! cannonical approach.
  template <class T0, class T1>
  void indirect(Eigen::ArrayBase<T1> const &coeffs, Eigen::ArrayBase<T0> &&signal) const {
    indirect(coeffs, signal);
  }

  //! Number of levels over which to do transform
  t_uint max_levels() const {
    auto cmp = [](Wavelet const &a, Wavelet const &b) { return a.levels() < b.levels(); };
    return std::max_element(begin(), end(), cmp)->levels();
  }

  //! Adds a wavelet of specific type
  void emplace_back(std::string const &name, t_uint nlevels) {
    std::vector<Wavelet>::emplace_back(factory(name, nlevels));
  }
};

#define SOPT_WAVELET_ERROR_MACRO(INPUT)                                                            \
  if(INPUT.rows() % (1u << max_levels()) != 0)                                                     \
    throw std::length_error("Inconsistent number of columns and wavelet levels");                  \
  else if(INPUT.cols() != 1 and INPUT.cols() % (1u << max_levels()))                               \
    throw std::length_error("Inconsistent number of rows and wavelet levels");

template <class T0, class T1>
void SARA::direct(Eigen::ArrayBase<T1> &coeffs, Eigen::ArrayBase<T0> const &signal) const {
  SOPT_WAVELET_ERROR_MACRO(signal);
  if(coeffs.rows() != signal.rows() or coeffs.cols() != signal.cols() * static_cast<t_int>(size()))
    coeffs.derived().resize(signal.rows(), signal.cols() * size());
  if(coeffs.rows() != signal.rows() or coeffs.cols() != signal.cols() * static_cast<t_int>(size()))
    throw std::length_error("Incorrect size for output matrix(or could not resize)");
  auto const Ncols = signal.cols();
#ifndef SOPT_OPENMP
  SOPT_TRACE("Calling direct sara without threads");
  for(size_type i(0); i < size(); ++i)
    at(i).direct(coeffs.leftCols((i + 1) * Ncols).rightCols(Ncols), signal);
#else
#pragma omp parallel
  {
    if(omp_get_thread_num() == 0) {
      SOPT_TRACE("Calling direct sara with {} threads of {}", omp_get_num_threads(),
                 omp_get_max_threads());
    }
#pragma omp for
    for(size_type i = 0; i < size(); ++i)
      at(i).direct(coeffs.leftCols((i + 1) * Ncols).rightCols(Ncols), signal);
#endif
}
coeffs /= std::sqrt(size());
}

template <class T0, class T1>
void SARA::indirect(Eigen::ArrayBase<T1> const &coeffs, Eigen::ArrayBase<T0> &signal) const {
  SOPT_WAVELET_ERROR_MACRO(coeffs);
  if(coeffs.cols() % size() != 0)
    throw std::length_error(
        "Columns of coefficient matrix and number of wavelets are inconsistent");
  if(coeffs.rows() != signal.rows() or coeffs.cols() != signal.cols() * static_cast<t_int>(size()))
    signal.derived().resize(coeffs.rows(), coeffs.cols() / size());
  if(coeffs.rows() != signal.rows() or coeffs.cols() != signal.cols() * static_cast<t_int>(size()))
    throw std::length_error("Incorrect size for output matrix(or could not resize)");

  auto const Ncols = signal.cols();
#ifndef SOPT_OPENMP
  SOPT_TRACE("Calling indirect sara without threads");
  signal = front().indirect(coeffs.leftCols(Ncols).rightCols(Ncols));
  for(size_type i(1); i < size(); ++i)
    signal += at(i).indirect(coeffs.leftCols((i + 1) * Ncols).rightCols(Ncols));
#else
  signal.fill(0);
#pragma omp parallel
  {
    if(omp_get_thread_num() == 0) {
      SOPT_TRACE("Calling indirect sara with {} threads of {}", omp_get_num_threads(),
                 omp_get_max_threads());
    }
    Image<typename T0::Scalar> reductor = Image<typename T0::Scalar>::Zero(signal.rows(), Ncols);
#pragma omp for
    for(size_type i = 0; i < size(); ++i)
      reductor += at(i).indirect(coeffs.leftCols((i + 1) * Ncols).rightCols(Ncols));
#pragma omp critical
    signal += reductor;
  }
#endif
  signal /= std::sqrt(size());
}

#undef SOPT_WAVELET_ERROR_MACRO

template <class T0>
typename T0::PlainObject SARA::indirect(Eigen::ArrayBase<T0> const &coeffs) const {
  typedef decltype(this->front().indirect(coeffs)) t_Output;
  t_Output signal = t_Output::Zero(coeffs.rows(), coeffs.cols() / size());
  (*this).indirect(coeffs, signal);
  return signal;
}

template <class T0>
typename T0::PlainObject SARA::direct(Eigen::ArrayBase<T0> const &signal) const {
  typedef decltype(this->front().direct(signal)) t_Output;
  t_Output result = t_Output::Zero(signal.rows(), signal.cols() * size());
  (*this).direct(result, signal);
  return result;
}
}
}
#endif
