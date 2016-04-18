#include <catch.hpp>
#include <random>

#include <Eigen/Dense>

#include "sopt/l1_padmm.h"
#include "sopt/padmm.h"
#include "sopt/proximal.h"
#include "sopt/types.h"

sopt::t_int random_integer(sopt::t_int min, sopt::t_int max) {
  extern std::unique_ptr<std::mt19937_64> mersenne;
  std::uniform_int_distribution<sopt::t_int> uniform_dist(min, max);
  return uniform_dist(*mersenne);
};

typedef sopt::t_real Scalar;
typedef sopt::Vector<Scalar> t_Vector;
typedef sopt::Matrix<Scalar> t_Matrix;

auto const N = 5;

TEST_CASE("Proximal ADMM with ||x - x0||_2 functions", "[padmm][integration]") {
  using namespace sopt;
  t_Vector const target0 = t_Vector::Random(N);
  t_Vector const target1 = t_Vector::Random(N) * 4;
  auto const g0 = proximal::translate(proximal::EuclidianNorm(), -target0);
  auto const g1 = proximal::translate(proximal::EuclidianNorm(), -target1);

  t_Matrix const mId = -t_Matrix::Identity(N, N);

  t_Vector const translation = t_Vector::Ones(N) * 5;
  auto const padmm = algorithm::ProximalADMM<Scalar>(g0, g1, t_Vector::Zero(N))
    .Phi(mId).itermax(3000).gamma(0.01);
  auto const result = padmm(t_Vector::Zero(N));

  t_Vector const segment = (target1 - target0).normalized();
  t_real const alpha = (result.x - target0).transpose() * segment;

  CHECK((target1 - target0).transpose() * segment >= alpha);
  CHECK(alpha >= 0e0);
  CAPTURE(segment.transpose());
  CAPTURE((result.x - target0).transpose());
  CAPTURE((result.x - target1).transpose());
  CHECK((result.x - target0 - alpha * segment).stableNorm() < 1e-8);
}

template <class T>
struct is_l1_proximal_ref : public std::is_same<sopt::algorithm::L1ProximalADMM<double> &, T> {};
TEST_CASE("Check type returned on setting variables") {
  // Yeah, could be static asserts
  using namespace sopt;
  using namespace sopt::algorithm;
  L1ProximalADMM<double> admm(Vector<double>::Zero(0));
  CHECK(is_l1_proximal_ref<decltype(admm.itermax(500))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.gamma(1e-1))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.relative_variation(5e-4))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.l2ball_proximal_epsilon(1e-4))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.tight_frame(false))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.l1_proximal_tolerance(1e-2))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.l1_proximal_nu(1))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.l1_proximal_itermax(50))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.l1_proximal_positivity_constraint(true))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.l1_proximal_real_constraint(true))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.residual_convergence(1.001))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.lagrange_update_scale(0.9))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.nu(1e0))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.target(Vector<double>::Zero(0)))>::value);
  typedef ConvergenceFunction<double> ConvFunc;
  CHECK(is_l1_proximal_ref<decltype(admm.is_converged(std::declval<ConvFunc>()))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.is_converged(std::declval<ConvFunc &>()))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.is_converged(std::declval<ConvFunc &&>()))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.is_converged(std::declval<ConvFunc const &>()))>::value);
  typedef LinearTransform<Vector<double>> LinTrans;
  CHECK(is_l1_proximal_ref<decltype(admm.Phi(linear_transform_identity<double>()))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.Phi(std::declval<LinTrans>()))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.Phi(std::declval<LinTrans &&>()))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.Phi(std::declval<LinTrans &>()))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.Phi(std::declval<LinTrans const &>()))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.Psi(linear_transform_identity<double>()))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.Psi(std::declval<LinTrans>()))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.Psi(std::declval<LinTrans &&>()))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.Psi(std::declval<LinTrans &>()))>::value);
  CHECK(is_l1_proximal_ref<decltype(admm.Psi(std::declval<LinTrans const &>()))>::value);
}
