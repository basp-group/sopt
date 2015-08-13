#ifndef SOPT_LOGGING_ENABLED_H
#define SOPT_LOGGING_ENABLED_H

#include <spdlog/spdlog.h>
namespace sopt { namespace logging {
  //! Name of the sopt logger
  const std::string name_prefix = "sopt";

  //! \brief Initializes a logger.
  //! \details Logger only exists as long as return is kept alive.
  inline std::shared_ptr<spdlog::logger> initialize(std::string const &name="") {
    return spdlog::stdout_logger_mt(name_prefix + name);
  }

  //! Returns shared pointer to logger or null if it does not exist
  inline std::shared_ptr<spdlog::logger> get(std::string const &name="") {
    return spdlog::get(name_prefix + name);
  }
}}

//! \macro For internal use only
#define SOPT_LOG_(NAME, TYPE, ...)                                                  \
  if(auto sopt_logging_ ## __func__ ## _ ## __LINE__ = sopt::logging::get(NAME))    \
     sopt_logging_ ## __func__ ## _ ## __LINE__->TYPE(__VA_ARGS__)

#endif
