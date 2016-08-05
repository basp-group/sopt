#ifndef SOPT_CPP_CONFIG_H
#define SOPT_CPP_CONFIG_H

//! Problems with using and constructors
#cmakedefine SOPT_HAS_USING
#ifndef SOPT_HAS_USING
#define SOPT_HAS_NOT_USING
#endif

//! True if using OPENMP
#cmakedefine SOPT_OPENMP

//! Macro to start logging or not
#cmakedefine SOPT_DO_LOGGING

#include <string>
#include <tuple>

namespace sopt {
//! Returns library version
inline std::string version() { return "@Sopt_VERSION@"; }
//! Returns library version
inline constexpr std::tuple<uint8_t, uint8_t, uint8_t> version_tuple() {
  return std::tuple<uint8_t, uint8_t, uint8_t>(
      @Sopt_VERSION_MAJOR@, @Sopt_VERSION_MINOR@, @Sopt_VERSION_PATCH@);
}
//! Returns library git reference, if known
inline std::string gitref() { return "@Sopt_GITREF@"; }
//! Default logging level
inline std::string default_logging_level() { return "@SOPT_TEST_LOG_LEVEL@"; }
//! Default logger name
inline std::string default_logger_name() { return "@SOPT_LOGGER_NAME@"; }
//! Wether to add color to the logger
inline constexpr bool color_logger() { return @SOPT_COLOR_LOGGING@; }
//! Number of threads used during testing
inline constexpr std::size_t number_of_threads_in_tests() { return @SOPT_DEFAULT_OPENMP_THREADS@; }
}

#endif
