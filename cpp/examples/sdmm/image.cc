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

  if(argc != 2 and argc != 3) {
    std::cout << "Expects one or two arguments:\n"
                 "- path to the image to clean (or name of standard SOPT image)\n"
                 "- optional path to output image\n";
    exit(0);
  }

  // Initializes and sets logger (if compiled with logging)
  // See set_level function for levels.
  sopt::logging::initialize();
  sopt::logging::set_level(SOPT_TEST_DEBUG_LEVEL);

  // Read input file - standard data file or full path to a tiff file
  auto const image = sopt::read_standard_tiff(argv[1]);

  SOPT_TRACE("Initializing SARA");
  sopt::wavelets::SARA const sara = {{"DB4", 5}}; //, {"DB8", 2}};
  auto const analysis = [&image, &sara](t_Vector &out, t_Vector const &x) -> void {
    // Maps from vectors (SDMM input) to matrices (images)
    auto out_mat = t_Array::Map(out.data(), image.rows(), image.cols() * sara.size());
    auto const x_mat = t_Array::Map(x.data(), image.rows(), image.cols());
    out_mat = sara.direct(x_mat);
  };
  auto const synthesis = [&image, &sara](t_Vector &out, t_Vector const &x) {
    // Maps from vectors (SDMM input) to matrices (images)
    auto out_mat = t_Array::Map(out.data(), image.rows(), image.cols());
    auto const x_mat = t_Array::Map(x.data(), image.rows(), image.cols() * sara.size());
    out_mat = sara.indirect(x_mat);
  };

  SOPT_TRACE("Initializing Proximals");
  // Proximal functions
  auto prox_g0 = [](t_Vector &out, Scalar gamma, t_Vector const &x) {
    out = sopt::proximal::l1_norm(gamma, x);
  };
  auto prox_g1 = sopt::proximal::translate(
      // Proximal for indicator function of L2 ball
      [](t_Vector &out, Scalar, t_Vector const & x) {
        auto const epsilon = 1e-4;
        auto const norm = x.stableNorm();
        out = x * (norm < epsilon ? 1e0: epsilon / norm);
      },
      // SDMM expects vectors as input, not matrices
      t_Vector::Map(image.data(), image.size())
  );
  auto prox_g2 = [](t_Vector &out, Scalar, t_Vector const & x) {
    out = sopt::positive_quadrant(x);
  };

  SOPT_TRACE("Initializing sensing operator");
  auto const Nr = image.size() * sara.size(), Ns = Nr / 100;
  auto sensing = sopt::Sampling(Ns, Nr).as_linear_transform<Scalar>();

  SOPT_TRACE("Initializing convergence function");
  auto relvar = sopt::RelativeVariation<Scalar>(1e-6);
  auto convergence = [&relvar](sopt::algorithm::SDMM<Scalar> const&, t_Vector const &x) {
    return relvar(x);
  };

  SOPT_TRACE("Initializing SDMM");
  // Now we can create the sdmm convex minimizer
  // Its parameters are set by calling member functions with appropriate names.
  auto sdmm = sopt::algorithm::SDMM<Scalar>()
    .itermax(500) // maximum number of iterations
    .gamma(1)
    .conjugate_gradient(std::numeric_limits<sopt::t_uint>::max(), 1e-3)
    .is_converged(convergence)
    // Any number of (proximal g_i, L_i) pairs can be added
    // ||Psi^dagger x||_1
    .append(prox_g0, synthesis, analysis)
    // ||y - A x|| < epsilon
    .append(prox_g1, sensing)
    // x in positive quadrant
    .append(prox_g2);

  SOPT_TRACE("Allocating result vector");
  t_Vector result = t_Vector::Zero(image.size());
  SOPT_DEBUG("Starting SDMM");
  auto const diagnostic = sdmm(result);
  SOPT_DEBUG("SDMM returned {}", diagnostic.good);

  // diagnostic should tell us the function converged
  // it also contains diagnostic.niters - the number of iterations, and cg_diagnostic - the
  // diagnostic from the last call to the conjugate gradient.
  if(not diagnostic.good)
    throw std::runtime_error("Did not converge!");

  SOPT_INFO("SOPT-SDMM converged in {} iterations", diagnostic.niters);
  if(argc == 3)
    sopt::write_tiff(t_Matrix::Map(result.data(), image.rows(), image.cols()), argv[2]);

  return 0;
}
