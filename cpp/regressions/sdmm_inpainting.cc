#include <complex>
#include "catch.hpp"
#include <iostream>
#include <iomanip>

#include "sopt/sdmm.h"
#include "sopt/logging.h"
#include "sopt/sampling.h"
#include "sopt/wavelets.h"
#include "sopt/relative_variation.h"
#include "sopt/proximal.h"
#include "tools_for_tests/tiffwrappers.h"
#include "tools_for_tests/directories.h"

typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> t_Vector;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> t_Matrix;
std::string const filename = "cameraman256.tiff";
std::string const outdir = sopt::notinstalled::output_directory() + "/sdmm/regressions/";
std::string const outfile = "inpainting.tiff";

t_Vector target(sopt::LinearTransform<t_Vector> const &sampling, sopt::t_rMatrix const &image) {
  return sampling * t_Vector::Map(image.data(), image.size());
}

Scalar sigma(sopt::LinearTransform<t_Vector> const &sampling, sopt::t_rMatrix const &image) {
  auto const snr = 30.0;
  auto const y0 = target(sampling, image);
  return y0.stableNorm() / std::sqrt(y0.size()) * std::pow(10.0, -(snr / 20.0));
}

t_Vector dirty(sopt::LinearTransform<t_Vector> const &sampling, sopt::t_rMatrix const &image) {
  using namespace sopt;

  std::mt19937 gen((std::random_device())());
  // values near the mean are the most likely
  // standard deviation affects the dispersion of generated values from the mean
  auto const y0 = target(sampling, image);
  std::normal_distribution<> gaussian_dist(0, sigma(sampling, image));
  t_Vector y(y0.size());
  for (t_int i = 0; i < y0.size(); i++)
    y(i) = y0(i) + gaussian_dist(gen);

  t_Vector dirty = sampling.adjoint() * y;
  assert(dirty.size() == image.size());
  notinstalled::write_tiff(
      t_Matrix::Map(dirty.data(), image.rows(), image.cols()),
      outdir + "dirty_" + outfile
  );
  return y;
}

sopt::algorithm::SDMM<Scalar> create_sdmm(
    sopt::LinearTransform<t_Vector> const &sampling,
    sopt::t_rMatrix const &image,
    t_Vector const &y) {
  using namespace sopt;

  auto const y0 = target(sampling, image);
  auto const nmeasure = y0.size();
  auto epsilon = std::sqrt(nmeasure + 2*std::sqrt(nmeasure)) * sigma(sampling, y0);
  auto prox_l2ball = proximal::translate(proximal::L2Ball<Scalar>(epsilon), -y);

  auto const wavelet = wavelets::factory("DB8", 4);
  auto const psi = linear_transform<Scalar>(wavelet, image.rows(), image.cols());

  auto relvar = RelativeVariation<Scalar>(1e-2);
  auto convergence = [&y, &sampling, &psi, &relvar](
      algorithm::SDMM<Scalar> const&, t_Vector const &x) {
    INFO("||x - y||_2: " << (y - sampling * x).stableNorm());
    INFO("||Psi^Tx||_1: " << l1_norm(psi.adjoint() * x));
    INFO("||abs(x) - x||_2: " << (x.array().abs().matrix() - x).stableNorm());
    return relvar(x);
  };

  // Now we can create the sdmm convex minimizer
  // Its parameters are set by calling member functions with appropriate names.
  return algorithm::SDMM<Scalar>()
    .itermax(500) // maximum number of iterations
    .gamma(0.1)
    .conjugate_gradient(300, 1e-8)
    .is_converged(convergence)
    // Any number of (proximal g_i, L_i) pairs can be added
    // ||Psi^dagger x||_1
    .append(proximal::l1_norm<Scalar>, psi.adjoint(), psi)
    // ||y - A x|| < epsilon
    .append(prox_l2ball, sampling)
    // x in positive quadrant
    .append(proximal::positive_quadrant<Scalar>);
}

TEST_CASE("Compare SDMMS", "") {
  using namespace sopt;

  // Read image and create target vector y
  t_rMatrix const image = notinstalled::read_standard_tiff(filename);
  t_uint const nmeasure = 0.5 * image.size();
  auto const sampling = linear_transform<Scalar>(Sampling(image.size(), nmeasure));
  auto const y = dirty(sampling, image);


  // Create c++ SDMM
  t_Vector result(image.size());
  auto const sdmm = ::create_sdmm(sampling, image, y);

  // Run sdmm
  auto const diagnostic = sdmm(result, t_Vector::Zero(image.size()));
  CHECK(diagnostic.good);

  notinstalled::write_tiff(
    t_Matrix::Map(result.data(), image.rows(), image.cols()),
    outdir + "result_" + outfile
  );
}
