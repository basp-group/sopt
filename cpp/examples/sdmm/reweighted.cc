#include <algorithm>
#include <exception>
#include <functional>
#include <random>
#include <vector>
#include <iostream>

#include <sopt/logging.h>
#include <sopt/maths.h>
#include <sopt/positive_quadrant.h>
#include <sopt/relative_variation.h>
#include <sopt/reweighted.h>
#include <sopt/sampling.h>
#include <sopt/sdmm.h>
#include <sopt/types.h>
#include <sopt/utilities.h>
#include <sopt/wavelets.h>
// This header is not part of the installed sopt interface
// It is only present in tests
#include <tools_for_tests/directories.h>
#include <tools_for_tests/tiffwrappers.h>

// \min_{x} ||W_j\Psi^Tx||_1 \quad \mbox{s.t.} \quad ||y - Ax||_2 < \epsilon and x \geq 0
// with W_j = ||\Psi^Tx_{j-1}||_1
// By iterating this algorithm, we can approximate L0 from L1.
int main(int argc, char const **argv) {
  // Some typedefs for simplicity
  typedef double Scalar;
  // Column vector - linear algebra - A * x is a matrix-vector multiplication
  // type expected by SDMM
  typedef sopt::Vector<Scalar> Vector;
  // Matrix - linear algebra - A * x is a matrix-vector multiplication
  // type expected by SDMM
  typedef sopt::Matrix<Scalar> Matrix;
  // Image - 2D array - A * x is a coefficient-wise multiplication
  // Type expected by wavelets and image write/read functions
  typedef sopt::Image<Scalar> Image;

  std::string const input = argc >= 2 ? argv[1] : "cameraman256";
  std::string const output = argc == 3 ? argv[2] : "none";
  if(argc > 3) {
    std::cout << "Usage:\n"
                 "$ "
              << argv[0] << " [input [output]]\n\n"
                            "- input: path to the image to clean (or name of standard SOPT image)\n"
                            "- output: filename pattern for output image\n";
    exit(0);
  }
  // Set up random numbers for C and C++
  auto const seed = std::time(0);
  std::srand((unsigned int)seed);
  std::mt19937 mersenne(std::time(0));

  // Initializes and sets logger (if compiled with logging)
  // See set_level function for levels.
  sopt::logging::initialize();

  SOPT_TRACE("Read input file {}", input);
  Image const image = sopt::notinstalled::read_standard_tiff(input);

  SOPT_TRACE("Initializing sensing operator");
  sopt::t_uint nmeasure = 0.33 * image.size();
  auto const sampling
      = sopt::linear_transform<Scalar>(sopt::Sampling(image.size(), nmeasure, mersenne));

  SOPT_TRACE("Initializing wavelets");
  auto const wavelet = sopt::wavelets::factory("DB4", 4);
  auto const psi = sopt::linear_transform<Scalar>(wavelet, image.rows(), image.cols());

  SOPT_TRACE("Computing sdmm parameters");
  Vector const y0 = sampling * Vector::Map(image.data(), image.size());
  auto const snr = 30.0;
  auto const sigma = y0.stableNorm() / std::sqrt(y0.size()) * std::pow(10.0, -(snr / 20.0));
  auto const epsilon = std::sqrt(nmeasure + 2 * std::sqrt(y0.size())) * sigma;

  SOPT_TRACE("Create dirty vector");
  std::normal_distribution<> gaussian_dist(0, sigma);
  Vector y(y0.size());
  for(sopt::t_int i = 0; i < y0.size(); i++)
    y(i) = y0(i) + gaussian_dist(mersenne);
  // Write dirty imagte to file
  if(output != "none") {
    Vector const dirty = sampling.adjoint() * y;
    sopt::utilities::write_tiff(Matrix::Map(dirty.data(), image.rows(), image.cols()),
                                "dirty_" + output + ".tiff");
  }

  SOPT_TRACE("Initializing convergence function");
  auto relvar = sopt::RelativeVariation<Scalar>(5e-2);
  auto convergence = [&y, &sampling, &psi, &relvar](sopt::Vector<Scalar> const &x) -> bool {
    SOPT_INFO("||x - y||_2: {}", (y - sampling * x).stableNorm());
    SOPT_INFO("||Psi^Tx||_1: {}", sopt::l1_norm(psi.adjoint() * x));
    SOPT_INFO("||abs(x) - x||_2: {}", (x.array().abs().matrix() - x).stableNorm());
    return relvar(x);
  };

  SOPT_TRACE("Creating SDMM Functor");
  auto const sdmm
      = sopt::algorithm::SDMM<Scalar>()
            .itermax(3000)
            .gamma(0.1)
            .conjugate_gradient(200, 1e-8)
            .is_converged(convergence)
            // Any number of (proximal g_i, L_i) pairs can be added
            // ||Psi^dagger x||_1
            .append(sopt::proximal::l1_norm<Scalar>, psi.adjoint(), psi)
            // ||y - A x|| < epsilon
            .append(sopt::proximal::translate(sopt::proximal::L2Ball<Scalar>(epsilon), -y),
                    sampling)
            // x in positive quadrant
            .append(sopt::proximal::positive_quadrant<Scalar>);

  SOPT_TRACE("Creating the reweighted algorithm");
  auto const posq = positive_quadrant(sdmm);
  typedef std::remove_const<decltype(posq)>::type t_PosQuadSDMM;
  auto const min_delta = sigma * std::sqrt(y.size()) / std::sqrt(8 * image.size());
  // Sets weight after each sdmm iteration.
  // In practice, this means replacing the proximal of the l1 objective function.
  auto set_weights = [](t_PosQuadSDMM &sdmm, Vector const &weights) {
    sdmm.algorithm().proximals(0) = [weights](Vector &out, Scalar gamma, Vector const &x) {
      out = sopt::soft_threshhold(x, gamma * weights);
    };
  };
  auto call_PsiT
      = [&psi](t_PosQuadSDMM const &, Vector const &x) -> Vector { return psi.adjoint() * x; };
  auto const reweighted = sopt::algorithm::reweighted(posq, set_weights, call_PsiT)
                              .itermax(5)
                              .min_delta(min_delta)
                              .is_converged(sopt::RelativeVariation<Scalar>(1e-3));

  SOPT_TRACE("Computing warm-start SDMM");
  auto warm_start = sdmm(Vector::Zero(image.size()));
  warm_start.x = sopt::positive_quadrant(warm_start.x);
  SOPT_TRACE("SDMM returned {}", warm_start.good);

  SOPT_TRACE("Computing warm-start SDMM");
  auto const result = reweighted(warm_start);

  // result should tell us the function converged
  // it also contains result.niters - the number of iterations, and cg_diagnostic - the
  // result from the last call to the conjugate gradient.
  if(not result.good)
    throw std::runtime_error("Did not converge!");

  SOPT_INFO("SOPT-SDMM converged in {} iterations", result.niters);
  if(output != "none")
    sopt::utilities::write_tiff(Matrix::Map(result.algo.x.data(), image.rows(), image.cols()),
                                output + ".tiff");

  return 0;
}
