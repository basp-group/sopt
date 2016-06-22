#include <catch.hpp>
#include <complex>
#include <iomanip>
#include <random>
#include <type_traits>

#include "sopt/logging.h"
#include "sopt/proximal.h"
#include "sopt/relative_variation.h"
#include "sopt/sampling.h"
#include "sopt/sdmm.h"
#include "sopt/wavelets.h"
#include "tools_for_tests/cdata.h"
#include "tools_for_tests/directories.h"
#include "tools_for_tests/inpainting.h"
#include "tools_for_tests/tiffwrappers.h"

typedef double Scalar;
typedef sopt::Vector<Scalar> t_Vector;
typedef sopt::Matrix<Scalar> t_Matrix;

sopt::algorithm::SDMM<Scalar> create_sdmm(sopt::LinearTransform<t_Vector> const &sampling,
                                          sopt::LinearTransform<t_Vector> const &psi,
                                          t_Vector const &y, t_Vector const &weights,
                                          sopt_l1_sdmmparam const &params) {

  using namespace sopt;
  auto weighted_l1_norm
      = [&weights](t_Vector &out, real_type<Scalar>::type gamma, t_Vector const &x) {
          assert(gamma > 1e-12);
          assert((weights.array() > 1e-12).all());
          out = soft_threshhold(x, gamma * weights);
        };
  return algorithm::SDMM<Scalar>()
      .itermax(params.max_iter)
      .gamma(params.gamma)
      .conjugate_gradient(params.cg_max_iter, params.cg_tol)
      .is_converged(RelativeVariation<Scalar>(params.rel_obj))
      .append(weighted_l1_norm, psi.adjoint(), psi)
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
  auto const y = dirty(sampling, image, *mersenne);

  sopt_l1_sdmmparam params = {
      0,                        // verbosity
      50,                       // max iter
      0.1,                      // gamma
      0.01,                     // relative change
      epsilon(sampling, image), // radius of the l2ball
      1e-3,                     // Relative tolerance on epsilon
      std::is_same<Scalar, sopt::real_type<Scalar>::type>::value ? 1 : 0, // real data
      200,                                                                // cg max iter
      1e-8,                                                               // cg tol
  };

  // Create c++ SDMM
  auto const wavelet = wavelets::factory("DB2", 2);
  auto const psi = linear_transform<Scalar>(wavelet, image.rows(), image.cols());

  // Create C bindings for C++ operators
  CData<Scalar> const sampling_data{image.size(), y.size(), sampling, 0, 0};
  CData<Scalar> const psi_data{image.size(), image.size(), psi, 0, 0};
  t_Vector const weights
      = t_Vector::Random((psi * t_Vector::Zero(image.size())).size()).array().abs();

  // Try increasing number of iterations and check output of c and c++ algorithms are the same
  for(t_uint i : {1, 2, 5, 10}) {
    SECTION(fmt::format("With {} iterations", i)) {
      t_Vector const input = t_Vector::Random(image.size());
      sopt_l1_sdmmparam c_params = params;
      c_params.max_iter = i;
      auto sdmm = ::create_sdmm(sampling, psi, y, weights, c_params);
      t_Vector cpp(image.size());
      sdmm(cpp, positive_quadrant(input));

      t_Vector c = input;
      sopt_l1_sdmm((void *)c.data(), c.size(), &direct_transform<Scalar>, (void **)&sampling_data,
                   &adjoint_transform<Scalar>, (void **)&sampling_data, &direct_transform<Scalar>,
                   (void **)&psi_data, &adjoint_transform<Scalar>, (void **)&psi_data, // synthesis
                   weights.size(), (void *)y.data(), y.size(), const_cast<double *>(weights.data()),
                   c_params);

      CHECK(cpp.isApprox(c));
    }
  };
}
