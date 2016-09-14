#ifndef SOPT_TIFF_WRAPPER_H
#define SOPT_TIFF_WRAPPER_H

#include "sopt/config.h"
#include <Eigen/Core>
#include "sopt/types.h"

namespace sopt {
namespace notinstalled {
//! Reads tiff image from sopt data directory if it exists
sopt::Image<> read_standard_tiff(std::string const &name);
}
} /* sopt::notinstalled */
#endif
