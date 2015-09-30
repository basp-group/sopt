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
#include <sopt/wavelets.h>
#include <sopt/sdmm.h>
#include <sopt/datadir.h>
#include "tiffwrappers.h"

// \min_{x} ||\Psi^Tx||_1 \quad \mbox{s.t.} \quad ||y - Ax||_2 < \epsilon and x \geq 0
int main(int argc, char const **argv) {
  // Some typedefs for simplicity
  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> t_Vector;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> t_Matrix;

  if(argc != 2 and argc != 3) {
    std::cout << "Expects one or two arguments:\n"
                 "- path to the image to clean (or name of standard SOPT image)\n"
                 "- optional path to output image\n";
    exit(0);
  }
  srand((unsigned int) 1);

  // Initializes and sets logger (if compiled with logging)
  // See set_level function for levels.
  sopt::logging::initialize();
  sopt::logging::set_level(SOPT_TEST_DEBUG_LEVEL);

  // Read input file - standard data file or full path to a tiff file
  sopt::t_rMatrix const image = sopt::read_standard_tiff(argv[1]);

  SOPT_TRACE("Initializing sensing operator");
  auto const sampling = sopt::linear_transform<Scalar>(sopt::Sampling(image.size(), image.size()));
  SOPT_DEBUG("rows: {}", sampling.rows(1));
  SOPT_DEBUG("rows: {}", sampling.rows(10));

  SOPT_TRACE("Create dirty sampled target");
  t_Vector const target = sampling * (
      t_Vector::Map(image.data(), image.size())
    + t_Vector::Random(image.size()) * 0.1
    - t_Vector::Ones(image.size()) * 0.05
  );

  SOPT_TRACE("Initializing Wavelet");
  auto const wavelet = sopt::wavelets::factory("DB4", 1);
  auto const psi = sopt::linear_transform<Scalar>(wavelet, image.rows(), image.cols());
  SOPT_DEBUG("Size of psi: {}, {}", (psi * t_Vector::Zero(image.size())).rows(), image.size());

  SOPT_TRACE("Initializing Proximals");
  // Proximal functions
  auto prox_l2ball = sopt::proximal::translate(sopt::proximal::L2Ball<Scalar>(1e-4), -target);

  SOPT_TRACE("Initializing convergence function");
  auto relvar = sopt::RelativeVariation<Scalar>(1e-6);
  auto convergence = [&target, &sampling, &psi, &relvar](
      sopt::algorithm::SDMM<Scalar> const&, t_Vector const &x) {
    SOPT_INFO("||x - target||_2: {}", (target - sampling * x).stableNorm());
    SOPT_INFO("||Psi^Tx||_1: {}", sopt::l1_norm(psi.adjoint() * x));
    SOPT_INFO("||abs(x) - x||_2: {}", (x.array().abs().matrix() - x).stableNorm());
    return relvar(x);
  };

  SOPT_TRACE("Initializing SDMM");
  // Now we can create the sdmm convex minimizer
  // Its parameters are set by calling member functions with appropriate names.
  auto sdmm = sopt::algorithm::SDMM<Scalar>()
    .itermax(5000) // maximum number of iterations
    .gamma(1)
    .conjugate_gradient(300, 1e-8)
    .is_converged(convergence)
    // Any number of (proximal g_i, L_i) pairs can be added
    // ||Psi^dagger x||_1
    .append(sopt::proximal::l1_norm<Scalar>, psi.adjoint(), psi)
    // ||y - A x|| < epsilon
    .append(prox_l2ball, sampling)
    // x in positive quadrant
    .append(sopt::proximal::positive_quadrant<Scalar>);

  SOPT_TRACE("Allocating result vector");
  t_Vector result(image.size());
  SOPT_TRACE("Starting SDMM");
  auto const diagnostic = sdmm(result, t_Vector::Zero(image.size()));
  SOPT_TRACE("SDMM returned {}", diagnostic.good);

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
