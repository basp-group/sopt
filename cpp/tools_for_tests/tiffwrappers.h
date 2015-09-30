#ifndef SOPT_TIFF_WRAPPER_H
#define SOPT_TIFF_WRAPPER_H

#include <Eigen/Core>

#include "sopt/types.h"

namespace sopt { namespace notinstalled {
  //! Reads tiff image
  sopt::t_rMatrix read_tiff(std::string const& name);
  //! Reads tiff image from sopt data directory if it exists
  sopt::t_rMatrix read_standard_tiff(std::string const& name);
  //! Writes a tiff greyscale file
  void write_tiff(sopt::t_rMatrix const & image, std::string const & filename);
}} /* sopt::notinstalled */
#endif
