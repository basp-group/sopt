#include <complex>
#include "catch.hpp"
#include <iostream>
#include <iomanip>
#include <type_traits>

#include "sopt/sdmm.h"
#include "sopt/logging.h"
#include "sopt/sampling.h"
#include "sopt/wavelets.h"
#include "sopt/relative_variation.h"
#include "sopt/proximal.h"
#include "tools_for_tests/tiffwrappers.h"
#include "tools_for_tests/directories.h"
#include "tools_for_tests/cdata.h"

extern "C" {
#include "sopt/sopt_l1.h"
}

typedef double Scalar;
typedef sopt::Vector<Scalar> t_Vector;
typedef sopt::Matrix<Scalar> t_Matrix;

t_Vector target(sopt::LinearTransform<t_Vector> const &sampling, sopt::Image<> const &image) {
  return sampling * t_Vector::Map(image.data(), image.size());
}

Scalar sigma(sopt::LinearTransform<t_Vector> const &sampling, sopt::Image<> const &image) {
  auto const snr = 30.0;
  auto const y0 = target(sampling, image);
  return y0.stableNorm() / std::sqrt(y0.size()) * std::pow(10.0, -(snr / 20.0));
}

t_Vector dirty(sopt::LinearTransform<t_Vector> const &sampling, sopt::Image<> const &image) {
  using namespace sopt;
  extern std::unique_ptr<std::mt19937_64> mersenne;

  // values near the mean are the most likely
  // standard deviation affects the dispersion of generated values from the mean
  auto const y0 = target(sampling, image);
  std::normal_distribution<> gaussian_dist(0, sigma(sampling, image));
  t_Vector y(y0.size());
  for (t_int i = 0; i < y0.size(); i++)
    y(i) = y0(i) + gaussian_dist(*mersenne);

  return y;
}

Scalar epsilon(sopt::LinearTransform<t_Vector> const &sampling, sopt::Image<> const &image) {
  auto const y0 = target(sampling, image);
  auto const nmeasure = y0.size();
  return std::sqrt(nmeasure + 2*std::sqrt(nmeasure)) * sigma(sampling, image);
}


sopt::algorithm::SDMM<Scalar> create_sdmm(
    sopt::LinearTransform<t_Vector> const &sampling,
    sopt::LinearTransform<t_Vector> const &psi,
    t_Vector const &y,
    sopt_l1_sdmmparam const &params) {

  using namespace sopt;
  return algorithm::SDMM<Scalar>()
    .itermax(params.max_iter)
    .gamma(params.gamma)
    .conjugate_gradient(params.cg_max_iter, params.cg_tol)
    .is_converged(RelativeVariation<Scalar>(params.rel_obj))
    .append(proximal::l1_norm<Scalar>, psi.adjoint(), psi)
    .append(proximal::translate(proximal::L2Ball<Scalar>(params.epsilon), -y), sampling)
    .append(proximal::positive_quadrant<Scalar>);
}

TEST_CASE("Compare SDMMS", "") {
  using namespace sopt;
  // Read image and create target vector y
  Image<> const image = notinstalled::read_standard_tiff("cameraman256");
  t_uint const nmeasure = 0.5 * image.size();
  extern std::unique_ptr<std::mt19937_64> mersenne;
  auto const sampling = linear_transform<Scalar>(Sampling(image.size(), nmeasure, *mersenne));
  auto const y = dirty(sampling, image);

  sopt_l1_sdmmparam params = {
    4,    // verbosity
    50,   // max iter
    0.1,  // gamma
    0.01, // relative change
    epsilon(sampling, image), // radius of the l2ball
    1e-3, // Relative tolerance on epsilon
    std::is_same<Scalar, sopt::real_type<Scalar>::type>::value ? 1: 0, // real data
    200,  // cg max iter
    1e-8, // cg tol
  };

  // Create c++ SDMM
  auto const wavelet = wavelets::factory("DB2", 2);
  auto const psi = linear_transform<Scalar>(wavelet, image.rows(), image.cols());

  // Create C bindings for C++ operators
  CData<Scalar> const sampling_data{image.size(), y.size(), sampling};
  CData<Scalar> const psi_data{image.size(), image.size(), psi};

  // Try increasing number of iterations and check output of c and c++ algorithms are the same
  for(t_uint i: {1, 2, 5, 10}) {
    SECTION(fmt::format("With {} iterations", i)) {
      sopt_l1_sdmmparam c_params = params;
      c_params.max_iter = i;
      auto sdmm = ::create_sdmm(sampling, psi, y, c_params);
      t_Vector cpp(image.size());
      auto const diagnostic = sdmm(cpp, t_Vector::Zero(image.size()));

      t_Vector c = t_Vector::Zero(image.size());
      t_Vector weights = t_Vector::Ones(image.size());
      sopt_l1_sdmm(
          (void*) c.data(), c.size(),
          &direct_transform<Scalar>, (void**)&sampling_data,
          &adjoint_transform<Scalar>, (void**)&sampling_data,
          &direct_transform<Scalar>, (void**)&psi_data,
          &adjoint_transform<Scalar>, (void**)&psi_data, // synthesis
          c.size(),
          (void*) y.data(), y.size(),
          weights.data(),
          c_params
      );

      CHECK(cpp.isApprox(c));
    }
  };
}
