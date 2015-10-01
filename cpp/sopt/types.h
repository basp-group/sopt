#ifndef BICO_TRAITS_H
#define BICO_TRAITS_H

#include <Eigen/Core>
#include <complex>

namespace sopt {

  //! Root of the type hierarchy for signed integers
  typedef int t_int;
  //! Root of the type hierarchy for unsigned integers
  typedef size_t t_uint;
  //! Root of the type hierarchy for real numbers
  typedef double t_real;
  //! Root of the type hierarchy for (real) complex numbers
  typedef std::complex<t_real> t_complex;


  template<class T = t_real>
    using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  template<class T = t_real>
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  template<class T = t_real>
    using Array = Eigen::Array<T, Eigen::Dynamic, 1>;
  template<class T = t_real>
    using Image = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
}
#endif

