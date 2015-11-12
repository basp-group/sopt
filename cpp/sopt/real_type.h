#ifndef SOPT_REAL_TYPE_H
#define SOPT_REAL_TYPE_H

#include <type_traits>
#include <complex>

namespace sopt {
namespace details {

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
}
//! Gets to the underlying real type
template<class T> using real_type = details::underlying_value_type<T>;
//! True if underlying type is complex
template<class T, class SP = void> struct is_complex : public std::false_type {};
//! True if underlying type is complex
template<class T> struct is_complex<std::complex<T>, void> : public std::true_type {};
} /* sopt  */
#endif
