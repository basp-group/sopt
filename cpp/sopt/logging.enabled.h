#ifndef SOPT_LOGGING_ENABLED_H
#define SOPT_LOGGING_ENABLED_H

#include "sopt/exception.h"
#include <spdlog/spdlog.h>

namespace sopt {
namespace logging {
//! Name of the sopt logger
const std::string name_prefix = "sopt";

//! \brief Initializes a logger.
//! \details Logger only exists as long as return is kept alive.
inline std::shared_ptr<spdlog::logger> initialize(std::string const &name = "") {
  return spdlog::stdout_logger_mt(name_prefix + name);
}

//! Returns shared pointer to logger or null if it does not exist
inline std::shared_ptr<spdlog::logger> get(std::string const &name = "") {
  return spdlog::get(name_prefix + name);
}

//! \brief Sets loggin level
//! \details Levels can be one of
//!     - "trace"
//!     - "debug"
//!     - "info"
//!     - "notice"
//!     - "warn"
//!     - "err"
//!     - "critical"
//!     - "alert"
//!     - "emerg"
//!     - "off"
inline void set_level(std::string const &level, std::string const &name = "") {
  auto const logger = get(name);
  if(not logger)
    SOPT_THROW("No logger by the name of ") << name << ".\n";
#define SOPT_MACRO(LEVEL)                                                                          \
  if(level == #LEVEL)                                                                              \
  logger->set_level(spdlog::level::LEVEL)
  SOPT_MACRO(trace);
  else SOPT_MACRO(debug);
  else SOPT_MACRO(info);
  else SOPT_MACRO(notice);
  else SOPT_MACRO(warn);
  else SOPT_MACRO(err);
  else SOPT_MACRO(critical);
  else SOPT_MACRO(alert);
  else SOPT_MACRO(emerg);
  else SOPT_MACRO(off);
#undef SOPT_MACRO
  else SOPT_THROW("Unknown logging level ") << level << "\n";
}

inline bool has_level(std::string const &level, std::string const &name = "") {
  auto const logger = get(name);
  if(not logger)
    return false;

#define SOPT_MACRO(LEVEL)                                                                          \
  if(level == #LEVEL)                                                                              \
  return logger->level() >= spdlog::level::LEVEL
  SOPT_MACRO(trace);
  else SOPT_MACRO(debug);
  else SOPT_MACRO(info);
  else SOPT_MACRO(notice);
  else SOPT_MACRO(warn);
  else SOPT_MACRO(err);
  else SOPT_MACRO(critical);
  else SOPT_MACRO(alert);
  else SOPT_MACRO(emerg);
  else SOPT_MACRO(off);
#undef SOPT_MACRO
  else SOPT_THROW("Unknown logging level ") << level << "\n";
}
}
}

//! \macro For internal use only
#define SOPT_LOG_(NAME, TYPE, ...)                                                                 \
  if(auto sopt_logging_##__func__##_##__LINE__ = sopt::logging::get(NAME))                         \
  sopt_logging_##__func__##_##__LINE__->TYPE(__VA_ARGS__)

#endif
