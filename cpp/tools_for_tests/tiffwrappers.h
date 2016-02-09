#ifndef SOPT_TIFF_WRAPPER_H
#define SOPT_TIFF_WRAPPER_H

#include "sopt/config.h"
#include <Eigen/Core>
#include "sopt/types.h"

namespace sopt {
namespace notinstalled {
//! Reads tiff image
sopt::Image<> read_tiff(std::string const &name);
//! Reads tiff image from sopt data directory if it exists
sopt::Image<> read_standard_tiff(std::string const &name);
//! Writes a tiff greyscale file
void write_tiff(sopt::Image<> const &image, std::string const &filename);
}
} /* sopt::notinstalled */
#endif
