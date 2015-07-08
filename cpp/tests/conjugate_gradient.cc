#include <random>
#include "catch.hpp"

#include "sopt/conjugate_gradient.h"

TEST_CASE("Conjugate gradient", "[cg]") {
  using namespace sopt;

  ConjugateGradient const cg(0, 1e-12);
  SECTION("Real valued") {
    auto const A = t_rMatrix::Random(10, 10).eval();
    auto const AtA = (A.transpose().matrix() * A.matrix()).eval();
    auto const expected = t_rVector::Random(A.rows()).eval();

    auto const actual = cg(AtA, (A.transpose().matrix() * expected.matrix()).eval());

    CHECK(actual.niters > 0);
    CHECK(actual.residual == Approx(0));
    CHECK((A.matrix() * actual.result).isApprox(expected.matrix(), 1e-7));
  }

  SECTION("Complex valued") {
    auto const A = t_cMatrix::Random(10, 10).eval();
    auto const AhA = (A.conjugate().transpose().matrix() * A.matrix()).eval();
    auto const expected = t_cVector::Random(A.rows()).eval();

    auto const actual = cg(AhA, (A.conjugate().transpose().matrix() * expected.matrix()).eval());

    CHECK(actual.niters > 0);
    CHECK(actual.residual == Approx(0));
    CHECK((A.matrix() * actual.result).isApprox(expected.matrix(), 1e-7));
  }
}
