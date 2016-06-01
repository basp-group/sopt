#ifndef SOPT_UTILITIES_H
#define SOPT_UTILITIES_H

#include "sopt/config.h"
#include <Eigen/Core>
#include "sopt/types.h"

namespace sopt {
namespace utilities {
//! Reads tiff image
sopt::Image<> read_tiff(std::string const &name);
//! Writes a tiff greyscale file
void write_tiff(Image<> const &image, std::string const &filename);
}
} /* sopt::utilities */
#endif

