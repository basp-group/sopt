#include <random>
#include "catch.hpp"

#include "sopt/conjugate_gradient.h"

TEST_CASE("Conjugate gradient", "[cg]") {
  using namespace sopt;

  auto const A = t_rMatrix::Random(10, 10).eval();
  auto const AtA = (A.transpose().matrix() * A.matrix()).eval();
  auto const expected = t_rVector::Random(A.rows()).eval();

  ConjugateGradient const cg;
  auto const actual = cg(AtA, (AtA * expected.matrix()).eval());

  CHECK(actual.niters > 0);
  CHECK(actual.residual == Approx(0));
}
