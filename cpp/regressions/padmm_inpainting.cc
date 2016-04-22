#include <catch.hpp>
#include <complex>
#include <iomanip>
#include <iostream>
#include <type_traits>

#include "sopt/l1_padmm.h"
#include "sopt/logging.h"
#include "sopt/proximal.h"
#include "sopt/relative_variation.h"
#include "sopt/sampling.h"
#include "sopt/wavelets.h"
#include "tools_for_tests/cdata.h"
#include "tools_for_tests/directories.h"
#include "tools_for_tests/inpainting.h"
#include "tools_for_tests/tiffwrappers.h"

typedef double Scalar;
typedef sopt::Vector<Scalar> t_Vector;
typedef sopt::Matrix<Scalar> t_Matrix;

sopt::algorithm::L1ProximalADMM<Scalar> create_admm(sopt::LinearTransform<t_Vector> const &phi,
                                                    sopt::LinearTransform<t_Vector> const &psi,
                                                    sopt_l1_param_padmm const &params,
                                                    t_Vector const &target) {

  using namespace sopt;
  return algorithm::L1ProximalADMM<Scalar>(target)
      .itermax(params.max_iter + 1)
      .gamma(params.gamma)
      .relative_variation(params.rel_obj)
      .l2ball_proximal_epsilon(params.epsilon)
      .tight_frame(params.paraml1.tight == 1)
      .l1_proximal_tolerance(params.paraml1.rel_obj)
      .l1_proximal_nu(params.paraml1.nu)
      .l1_proximal_itermax(params.paraml1.max_iter)
      .l1_proximal_positivity_constraint(params.paraml1.pos == 1)
      .l1_proximal_real_constraint(params.real_out)
      .residual_convergence(params.epsilon * params.epsilon_tol_scale)
      .lagrange_update_scale(params.lagrange_update_scale)
      .nu(params.nu)
      .Psi(psi)
      .Phi(phi)
      // just for show, 1 is the default value, so these calls do not do anything
      .l2ball_proximal_weights(Vector<t_real>::Ones(1))
      .l1_proximal_weights(Vector<t_real>::Ones(1));
}

TEST_CASE("Compare ADMM C++ and C", "") {
  using namespace sopt;
  // Read image and create target vector y
  Image<> const image = notinstalled::read_standard_tiff("cameraman256");
  t_uint const nmeasure = 0.5 * image.size();
  extern std::unique_ptr<std::mt19937_64> mersenne;
  auto const sampling = linear_transform<Scalar>(Sampling(image.size(), nmeasure, *mersenne));
  auto const y = dirty(sampling, image, *mersenne);

  sopt_l1_param_padmm params{
      0,                        // verbosity
      200,                      // max iter
      0.1,                      // gamma
      0.0005,                   // relative change
      epsilon(sampling, image), // radius of the l2ball
      1,                        // real out
      1,                        // real meas
      {
          0,    // verbose = 1;
          8,    // max_iter = 50;
          0.01, // rel_obj = 0.01;
          1,    // nu = 1;
          0,    // tight = 0;
          1,    // pos = 1;
      },
      1.001, // epsilon tol scale
      0.9,   // lagrange_update_scale
      1.0,   // nu
  };

  // Create c++ SDMM
  auto const wavelet = wavelets::factory("DB2", 2);
  auto const psi = linear_transform<Scalar>(wavelet, image.rows(), image.cols());

  // Create C bindings for C++ operators
  CData<Scalar> const sampling_data{image.size(), y.size(), sampling, 0, 0};
  CData<Scalar> const psi_data{image.size(), image.size(), psi, 0, 0};

  // Try increasing number of iterations and check output of c and c++ algorithms are the same
  for(t_uint i : {1, 5, 10}) {
    SECTION(fmt::format("With {} iterations", i)) {
      sopt_l1_param_padmm c_params = params;
      c_params.max_iter = i;
      auto admm = ::create_admm(sampling, psi, c_params, y);
      auto const cpp = admm();

      t_Vector c = t_Vector::Zero(image.size());
      t_Vector l1_weights = t_Vector::Ones(image.size());
      t_Vector l2_weights = t_Vector::Ones((sampling * image).size());
      sopt_l1_solver_padmm(
          (void *)c.data(), c.size(), &direct_transform<Scalar>, (void **)&sampling_data,
          &adjoint_transform<Scalar>, (void **)&sampling_data, &direct_transform<Scalar>,
          (void **)&psi_data, &adjoint_transform<Scalar>,
          (void **)&psi_data, // synthesis
          c.size(), (void *)y.data(), y.size(), l1_weights.data(), l2_weights.data(), c_params);

      CAPTURE(cpp.x.head(5).transpose());
      CAPTURE(c.head(5).transpose());
      CHECK(cpp.x.isApprox(c));
    }
  };
}
