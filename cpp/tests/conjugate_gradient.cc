#include <random>
#include "catch.hpp"

#include "sopt/conjugate_gradient.h"

TEST_CASE("Conjugate gradient", "[cg]") {
  using namespace sopt;

  ConjugateGradient const cg(std::numeric_limits<t_uint>::max(), 1e-12);
  SECTION("Real valued") {
    auto const A = Image<>::Random(10, 10).eval();
    auto const AtA = (A.transpose().matrix() * A.matrix()).eval();
    auto const expected = Array<>::Random(A.rows()).eval();

    auto const actual = cg(AtA, (A.transpose().matrix() * expected.matrix()).eval());

    CHECK(actual.niters > 0);
    CHECK(actual.residual == Approx(0));
    CAPTURE(actual.residual);
    CAPTURE((A.matrix() * actual.result).transpose());
    CAPTURE(expected.transpose());
    CHECK((A.matrix() * actual.result).isApprox(expected.matrix(), 1e-6));
  }

  SECTION("Complex valued") {
    auto const A = Image<t_complex>::Random(10, 10).eval();
    auto const AhA = (A.conjugate().transpose().matrix() * A.matrix()).eval();
    auto const expected = Array<t_complex>::Random(A.rows()).eval();

    auto const actual = cg(AhA, (A.conjugate().transpose().matrix() * expected.matrix()).eval());

    CHECK(actual.niters > 0);
    CHECK(actual.residual == Approx(0));
    CAPTURE(actual.residual);
    CAPTURE((A.matrix() * actual.result).transpose());
    CAPTURE(expected.transpose());
    CHECK((A.matrix() * actual.result).isApprox(expected.matrix(), 1e-6));
  }
}
