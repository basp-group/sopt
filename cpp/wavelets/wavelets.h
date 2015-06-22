#ifndef SOPT_WAVELETS_WAVELETS_H
#define SOPT_WAVELETS_WAVELETS_H

#include <iostream>
#include "wavelet_data.h"
#include "direct.h"
#include "indirect.h"
#include "types.h"

namespace sopt { namespace wavelets {

// Advance declaration so we can define the subsequent friend function
class Wavelet;

//! \brief Creates a wavelet transform object
Wavelet factory(std::string name="DB1", t_uint nlevels = 1);

//! Performs direct and indirect wavelet transforms
class Wavelet : public WaveletData
{
  friend Wavelet factory(std::string name, t_uint nlevels);
  protected:
    //! Should be called through factory function
    Wavelet(WaveletData const &c, t_uint nlevels) : WaveletData(c), levels_(nlevels) {}

  public:
    //! Destructor
    virtual ~Wavelet() {}

    // Temporary macros that checks constraints on input
#   define SOPT_WAVELET_MACRO_MULTIPLE(NAME)                                                      \
        if((NAME.rows() == 1 or NAME.cols() == 1)) {                                              \
          if(NAME.size() % (1 << levels()) != 0)                                                  \
            throw std::length_error("Size of " #NAME " must number a multiple of 2^levels or 1"); \
        } else if(NAME.rows() != 1 and NAME.rows() % (1 << levels()) != 0)                        \
          throw std::length_error("Rows of " #NAME " must number a multiple of 2^levels or 1");   \
        else if(NAME.cols() % (1 << levels()) != 0)                                               \
          throw std::length_error("Columns of " #NAME " must number a multiple of 2^levels");
#   define SOPT_WAVELET_MACRO_EQUAL_SIZE(A, B)                                                   \
        if(A.rows() != B.rows() or A.cols() != B.cols())                                         \
          throw std::length_error("Size of coefficients and signals must match");
    //! \brief Direct transform
    //! \param[in] signal: computes wavelet coefficients for this signal. Its size must be a
    //! multiple of $2^l$ where $l$ is the number of levels. Can be a matrix (2d-transform) or a
    //! column vector (1-d transform).
    //! \return wavelet coefficients
    //! \details Supports 1 and 2 dimensional tranforms for real and complex data.
    template<class T0>
      auto direct(Eigen::MatrixBase<T0> const &signal) const
      -> decltype(direct_transform(signal, 1, *this)) {
        SOPT_WAVELET_MACRO_MULTIPLE(signal);
        return direct_transform(signal, levels(), *this);
      }
    //! \brief Direct transform
    //! \param[inout] coefficients: Output wavelet coefficients. Must be of the same size and type
    //! as the input.
    //! \param[in] signal: computes wavelet coefficients for this signal. Its size must be a
    //! multiple of $2^l$ where $l$ is the number of levels. Can be a matrix (2d-transform) or a
    //! column vector
    //! (1-d transform).
    //! \details Supports 1 and 2 dimensional tranforms for real and complex data.
    template<class T0, class T1>
      auto direct(Eigen::MatrixBase<T1> & coefficients, Eigen::MatrixBase<T0> const &signal) const
      -> decltype(direct_transform(coefficients, signal, 1, *this)) {
        SOPT_WAVELET_MACRO_MULTIPLE(signal);
        SOPT_WAVELET_MACRO_EQUAL_SIZE(coefficients, signal);
        return direct_transform(coefficients, signal, levels(), *this);
      }
    //! \brief Direct transform
    //! \param[inout] coefficients: Output wavelet coefficients. Must be of the same size and type
    //! as the input.
    //! \param[in] signal: computes wavelet coefficients for this signal. Its size must be a
    //! multiple of $2^l$ where $l$ is the number of levels. Can be a matrix (2d-transform) or a
    //! column vector
    //! (1-d transform).
    //! \details Supports 1 and 2 dimensional tranforms for real and complex data. This version
    //! allows non-constant Eigen expressions to be passe on without the ugly `const_cast` of the
    //! cannonical approach.
    template<class T0, class T1>
      auto direct(Eigen::MatrixBase<T1> && coefficients, Eigen::MatrixBase<T0> const &signal) const
      -> decltype(direct_transform(coefficients, signal, 1, *this)) {
        SOPT_WAVELET_MACRO_MULTIPLE(signal);
        SOPT_WAVELET_MACRO_EQUAL_SIZE(coefficients, signal);
        return direct_transform(coefficients, signal, levels(), *this);
      }
    //! \brief Indirect transform
    //! \param[in] coefficients: Input wavelet coefficients. Its size must be a multiple of $2^l$
    //! where $l$ is the number of levels. Can be a matrix (2d-transform) or a column vector (1-d
    //! transform).
    //! \details Supports 1 and 2 dimensional tranforms for real and complex data.
    template<class T0>
      auto indirect(Eigen::MatrixBase<T0> const &coefficients) const
      -> decltype(indirect_transform(coefficients, 1, *this)) {
        SOPT_WAVELET_MACRO_MULTIPLE(coefficients);
        return indirect_transform(coefficients, levels(), *this);
      }
    //! \brief Indirect transform
    //! \param[in] coefficients: Input wavelet coefficients. Its size must be a multiple of $2^l$
    //! where $l$ is the number of levels. Can be a matrix (2d-transform) or a column vector (1-d
    //! \param[inout] signal: Reconstructed signal. Must be of the same size and type as the input.
    //! \details Supports 1 and 2 dimensional tranforms for real and complex data.
    template<class T0, class T1>
      auto indirect(Eigen::MatrixBase<T1> const & coefficients, Eigen::MatrixBase<T0> &signal) const
      -> decltype(indirect_transform(coefficients, signal, 1, *this)) {
        SOPT_WAVELET_MACRO_MULTIPLE(coefficients);
        SOPT_WAVELET_MACRO_EQUAL_SIZE(coefficients, signal);
        return indirect_transform(coefficients, signal, levels(), *this);
      }
    //! \brief Indirect transform
    //! \param[in] coefficients: Input wavelet coefficients. Its size must be a multiple of $2^l$
    //! where $l$ is the number of levels. Can be a matrix (2d-transform) or a column vector (1-d
    //! \param[inout] signal: Reconstructed signal. Must be of the same size and type as the input.
    //! \details Supports 1 and 2 dimensional tranforms for real and complex data.  This version
    //! allows non-constant Eigen expressions to be passe on without the ugly `const_cast` of the
    //! cannonical approach.
    template<class T0, class T1>
      auto indirect(Eigen::MatrixBase<T1> const & coeffs, Eigen::MatrixBase<T0> &&signal) const
      -> decltype(indirect_transform(coeffs, signal, 1, *this)) {
        SOPT_WAVELET_MACRO_MULTIPLE(coeffs);
        SOPT_WAVELET_MACRO_EQUAL_SIZE(coeffs, signal);
        return indirect_transform(coeffs, signal, levels(), *this);
      }
#   undef SOPT_WAVELET_MACRO_MULTIPLE
#   undef SOPT_WAVELET_MACRO_EQUAL_SIZE
    //! Number of levels over which to do transform
    t_uint levels() const { return levels_; }
    //! Sets number of levels over which to do transform
    void levels(t_uint l) { levels_ = l; }

    protected:
      //! Number of levels in the wavelet
      t_uint levels_;
};

}}
#endif
