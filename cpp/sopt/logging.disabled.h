#ifndef SOPT_LOGGING_DISABLED_H
#define SOPT_LOGGING_DISABLED_H

#include <memory>
#include <string>

namespace sopt {
namespace logging {
//! Name of the sopt logger
const std::string name_prefix = "sopt::";

inline std::shared_ptr<int> initialize(std::string const &) { return nullptr; }
inline std::shared_ptr<int> initialize() { return nullptr; }
inline std::shared_ptr<int> get(std::string const &) { return nullptr; }
inline std::shared_ptr<int> get() { return nullptr; }
inline void set_level(std::string const &, std::string const &){};
inline void set_level(std::string const &){};
inline bool has_level(std::string const &, std::string const &) { return false; }
}
}

//! \macro For internal use only
#define SOPT_LOG_(...)

#endif
