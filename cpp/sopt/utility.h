#ifndef SOPT_UTILITY_H
#define SOPT_UTILITY_H

#include <Eigen/Core>
#include <type_traits>
#include <algorithm>
#include <complex>

namespace sopt {

namespace details {
  //! Expression to create projection onto positive orthant
  template<class SCALAR> class ProjectPositiveQuadrant {
     public:
       SCALAR operator()(const SCALAR &value) const { return std::max(value, SCALAR(0)); }
  };
  //! Specialization for complex numbers
  template<class SCALAR> class ProjectPositiveQuadrant<std::complex<SCALAR>> {
     public:
       SCALAR operator()(SCALAR const &value) const {
         return std::max(value, SCALAR(0));
       }
       std::complex<SCALAR> operator()(std::complex<SCALAR> const &value) const {
         return std::complex<SCALAR>((*this)(value.real()), SCALAR(0));
       }
  };

  //! Expression to create projection onto positive orthant
  template<class SCALAR> class SoftThreshhold {
     public:
       typedef typename Eigen::NumTraits<SCALAR>::Real t_Threshhold;
       SoftThreshhold(t_Threshhold const &threshhold) : threshhold(threshhold) {
         if(threshhold < 0e0)
           throw std::domain_error("Threshhold must be negative");
       }
       SCALAR operator()(const SCALAR &value) const {
         auto const normalized = std::abs(value);
         return normalized < threshhold ? SCALAR(0): (value * (SCALAR(1) - threshhold/normalized));
       }
     private:
       t_Threshhold threshhold;
  };

  // Checks wether a type has contains a type "value_type"
  template<class T, class Enable=void> struct HasValueType {
      using Have = char[1];
      using HaveNot = char[2];

      struct Fallback {struct value_type {};};
      struct Derived: T, Fallback {};
      template<class U> static Have& test(typename U::value_type*);
      template<typename U> static HaveNot& test(U*);
    public:
      static constexpr bool value = sizeof(test<Derived>(nullptr)) == sizeof(HaveNot);
  };
  // Specialization for fundamental type that cannot be derived from
  template<class T>
    struct HasValueType<T, typename std::enable_if<std::is_fundamental<T>::value>::type>
        : std::false_type {};
}

//! Detects whether a class contains a value_type type
template<class T, bool = details::HasValueType<T>::value> class has_value_type;
template<class T> class has_value_type<T, true> : public std::true_type {};
template<class T> class has_value_type<T, false> : public std::false_type {};
//! Computes inner-most element type
template<class T, bool = has_value_type<T>::value> class underlying_value_type;
template<class T> class underlying_value_type<T, false> {
  public:
    typedef T type;
};
template<class T> class underlying_value_type<T, true> {
  public:
    typedef typename underlying_value_type<typename T::value_type>::type type;
};

//! Expression to create projection onto positive quadrant
template<class T>
Eigen::CwiseUnaryOp<const details::ProjectPositiveQuadrant<typename T::Scalar>, const T>
positive_quadrant(Eigen::DenseBase<T> const &input) {
  typedef details::ProjectPositiveQuadrant<typename T::Scalar> Projector;
  typedef Eigen::CwiseUnaryOp<const Projector, const T> UnaryOp;
  return UnaryOp(input.derived(), Projector());
}

//! Expression to create soft-threshhold
template<class T>
Eigen::CwiseUnaryOp<const details::SoftThreshhold<typename T::Scalar>, const T>
soft_threshhold(Eigen::DenseBase<T> const &input, typename T::Scalar const &threshhold) {
  typedef details::SoftThreshhold<typename T::Scalar> Threshhold;
  typedef Eigen::CwiseUnaryOp<const Threshhold, const T> UnaryOp;
  return UnaryOp(input.derived(), Threshhold(threshhold));
}


//! Computes weighted L1 norm
template<class T0, class T1>
  auto l1_norm(Eigen::ArrayBase<T0> const& input, Eigen::ArrayBase<T1> const &weight)
  -> decltype((input.cwiseAbs() * weight)(0)) { return (input.cwiseAbs() * weight).sum(); }
template<class T0, class T1>
  auto l1_norm(Eigen::MatrixBase<T0> const& input, Eigen::MatrixBase<T1> const &weight)
  -> decltype(l1_norm(input.array(), weight.array())) {
    return l1_norm(input.array(), weight.array());
  }
template<class T0, class T1>
  auto l1_norm(Eigen::MatrixBase<T0> const& input, Eigen::ArrayBase<T1> const &weight)
  -> decltype(l1_norm(input.array(), weight)) { return l1_norm(input.array(), weight); }
template<class T0, class T1>
  auto l1_norm(Eigen::ArrayBase<T0> const& input, Eigen::MatrixBase<T1> const &weight)
  -> decltype(l1_norm(input, weight.array())) { return l1_norm(input, weight.array()); }
//! Computes L1 norm
template<class T0>
  auto l1_norm(Eigen::ArrayBase<T0> const& input) -> decltype(input.abs()(0)) {
    return input.cwiseAbs().sum();
  }
template<class T0>
  auto l1_norm(Eigen::MatrixBase<T0> const& input) -> decltype(l1_norm(input.array())) {
    return l1_norm(input.array());
  }

} /* sopt */ 
#endif
