#include <random>
#include "catch.hpp"

#include <Eigen/Dense>
#include "sopt/proximal.h"
#include "sopt/padmm.h"

std::random_device rd;
std::default_random_engine rengine(rd());
sopt::t_int random_integer(sopt::t_int min, sopt::t_int max) {
  std::uniform_int_distribution<sopt::t_int> uniform_dist(min, max);
  return uniform_dist(rengine);
};

typedef double Scalar;
typedef sopt::Vector<Scalar> t_Vector;
typedef sopt::Matrix<Scalar> t_Matrix;

auto const N = 3;

TEST_CASE("Do nothing") {
  using namespace sopt;
  t_Vector const translation = t_Vector::Ones(N) * 5;
  t_Vector const input = t_Vector::Random(N).array() + 1e0;
  t_Matrix const Psi = t_Vector::Random(N, N).array();
  t_Vector output;
  auto const padmm = algorithm::PADMM<Scalar>().Psi(Psi).itermax(30).gamma(0.01);
  padmm(output, input);
}
