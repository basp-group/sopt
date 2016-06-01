#include <catch.hpp>
#include <complex>
#include <iomanip>
#include <random>
#include <type_traits>

#include "sopt/logging.h"
#include "sopt/positive_quadrant.h"
#include "sopt/proximal.h"
#include "sopt/relative_variation.h"
#include "sopt/reweighted.h"
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
                                          t_Vector const &y, sopt_l1_sdmmparam const &params) {

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

sopt::algorithm::Reweighted<sopt::algorithm::PositiveQuadrant<sopt::algorithm::SDMM<Scalar>>>
create_rwsdmm(sopt::algorithm::SDMM<Scalar> const &sdmm, sopt::LinearTransform<t_Vector> const &psi,
              sopt_l1_rwparam const &params) {
  using namespace sopt::algorithm;
  auto set_weights = [](PositiveQuadrant<SDMM<Scalar>> &sdmm, t_Vector const &weights) {
    sdmm.algorithm().proximals(0) = [weights](t_Vector &out, Scalar gamma, t_Vector const &x) {
      out = sopt::soft_threshhold(x, gamma * weights);
    };
  };
  auto call_PsiT = [&psi](PositiveQuadrant<SDMM<Scalar>> const &, t_Vector const &x) -> t_Vector {
    return psi.adjoint() * x;
  };
  auto const pq = positive_quadrant(sdmm);
  return reweighted(pq, set_weights, call_PsiT)
      .itermax(params.max_iter)
      .min_delta(params.sigma)
      .is_converged(sopt::RelativeVariation<Scalar>(params.rel_var));
}

TEST_CASE("Compare Reweighted SDMMS", "") {
  using namespace sopt;
  // Read image and create target vector y
  Image<> const image = notinstalled::read_standard_tiff("cameraman256");
  t_uint const nmeasure = 0.5 * image.size();
  extern std::unique_ptr<std::mt19937_64> mersenne;
  auto const sampling = linear_transform<Scalar>(Sampling(image.size(), nmeasure, *mersenne));
  auto const y = dirty(sampling, image, *mersenne);

  sopt_l1_sdmmparam params = {
      10,                       // verbosity
      2,                        // max iter
      0.1,                      // gamma
      0.01,                     // relative change
      epsilon(sampling, image), // radius of the l2ball
      1e-3,                     // Relative tolerance on epsilon
      std::is_same<Scalar, sopt::real_type<Scalar>::type>::value ? 1 : 0, // real data
      200,                                                                // cg max iter
      1e-8,                                                               // cg tol
  };

  sopt_l1_rwparam rwparam = {
      0, 5, 0.001, sigma(sampling, image) * std::sqrt(y.size()) / std::sqrt(8 * image.size()), 0};
  // Create c++ SDMM
  auto const wavelet = wavelets::factory("DB2", 2);
  auto const psi = linear_transform<Scalar>(wavelet, image.rows(), image.cols());

  // Create C bindings for C++ operators
  CData<Scalar> const sampling_data{image.size(), y.size(), sampling, 0, 0};
  CData<Scalar> const psi_data{image.size(), image.size(), psi, 0, 0};
  auto sdmm = ::create_sdmm(sampling, psi, y, params);
  auto warm_start = sdmm(t_Vector::Zero(image.size()));
  warm_start.x = positive_quadrant(warm_start.x);

  // Try increasing number of iterations and check output of c and c++ algorithms are the same
  for(t_uint i : {0, 1, 2, 5}) {
    SECTION(fmt::format("With {} iterations", i)) {
      sopt_l1_rwparam c_params = rwparam;
      c_params.max_iter = i;
      auto rwsdmm = ::create_rwsdmm(sdmm, psi, c_params);
      auto const cpp = rwsdmm(warm_start);

      CHECK(image.size() == (psi.adjoint() * t_Vector::Zero(image.size())).size());
      t_Vector c = t_Vector::Zero(image.size());
      t_Vector weights = t_Vector::Ones((psi.adjoint() * c).size());
      sopt_l1_rwsdmm((void *)c.data(), c.size(), &direct_transform<Scalar>, (void **)&sampling_data,
                     &adjoint_transform<Scalar>, (void **)&sampling_data, &direct_transform<Scalar>,
                     (void **)&psi_data, &adjoint_transform<Scalar>,
                     (void **)&psi_data, // synthesis
                     (psi.adjoint() * c).size(), (void *)y.data(), y.size(), params, c_params);

      CAPTURE(cpp.algo.x.head(25).tail(3).transpose());
      CAPTURE(c.head(25).tail(3).transpose());
      CAPTURE((cpp.algo.x - c).head(25).tail(3).transpose());
      CHECK(cpp.algo.x.isApprox(c));
    }
  };
}
