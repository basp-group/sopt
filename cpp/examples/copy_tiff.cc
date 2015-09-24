#include <exception>
#include <algorithm>
#include <vector>
#include <random>
#include <functional>

#include <sopt/logging.h>
#include <sopt/types.h>
#include <sopt/utility.h>
#include <sopt/sampling.h>
#include <sopt/relative_variation.h>
#include <wavelets/sara.h>
#include <sopt/sdmm.h>
#include <sopt/datadir.h>
#include "tiffwrappers.h"

// \min_{x} ||\Psi^\dagger x||_1 \quad \mbox{s.t.} \quad ||y - x||_2 < \epsilon and x \geq 0
int main(int argc, char const **argv) {
  // Some typedefs for simplicity
  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> t_Vector;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> t_Matrix;
  typedef Eigen::Array<t_Matrix::Scalar, Eigen::Dynamic, Eigen::Dynamic> t_Array;

  if(argc != 3) {
    std::cout << "Expects two arguments:\n"
                 "- path to the image to clean (or name of standard SOPT image)\n"
                 "- path to output image\n";
    exit(0);
  }

  // Initializes and sets logger (if compiled with logging)
  // See set_level function for levels.
  sopt::logging::initialize();
  sopt::logging::set_level(SOPT_TEST_DEBUG_LEVEL);

  // Read input file
  auto const image = sopt::read_standard_tiff(argv[1]);
  sopt::write_tiff(image, argv[2]);

  auto const reread = sopt::read_tiff(argv[2]);

  return 0;
}
