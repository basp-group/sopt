#include <random>
#include "catch.hpp"

#include <Eigen/Dense>
#include "sopt/proximal.h"
#include "sopt/padmm.h"
#include "sopt/types.h"

sopt::t_int random_integer(sopt::t_int min, sopt::t_int max) {
  extern std::unique_ptr<std::mt19937_64> mersenne;
  std::uniform_int_distribution<sopt::t_int> uniform_dist(min, max);
  return uniform_dist(*mersenne);
};

typedef sopt::t_real Scalar;
typedef sopt::Vector<Scalar> t_Vector;
typedef sopt::Matrix<Scalar> t_Matrix;

auto const N = 3;

TEST_CASE("Proximal ADMM with ||x - x0||_2 functions", "[padmm][integration]") {
  using namespace sopt;
  t_Vector const translation = t_Vector::Ones(N) * 5;
  t_Vector const input = t_Vector::Random(N).array() + 1e0;
  t_Matrix const Psi = t_Vector::Random(N, N).array();
  t_Vector output;
  auto const admm = algorithm::ProximalADMM<Scalar>().Psi(Psi).itermax(30).gamma(0.01);
  admm(output, input);
}
