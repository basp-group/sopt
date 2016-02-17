#include <fstream>

#include "sopt/types.h"
#include "sopt/utilities.h"
#include "tools_for_tests/directories.h"
#include "tools_for_tests/tiffwrappers.h"

namespace sopt {
namespace notinstalled {
Image<> read_standard_tiff(std::string const &name) {
  std::string const stdname = sopt::notinstalled::data_directory() + "/" + name + ".tiff";
  bool const is_std = std::ifstream(stdname).good();
  return sopt::utilities::read_tiff(is_std ? stdname : name);
}
}
} /* sopt::notinstalled  */
