#include <random>
#include "catch.hpp"

#include "sopt/utility.h"
#include "sopt/types.h"

TEST_CASE("Projector on positive quadrant", "[utility]") {
  using namespace sopt;

  SECTION("Real matrix") {
    t_rMatrix input = t_rMatrix::Ones(5, 5) + t_rMatrix::Random(5, 5) * 0.55;
    input(1, 1) *= -1; input(3, 2) *= -1;

    auto const expr = positive_quadrant(input);
    CAPTURE(input);
    CAPTURE(expr);
    CHECK(expr(1, 1) == Approx(0));
    CHECK(expr(3, 2) == Approx(0));

    auto value = expr.eval();
    CHECK(value(1, 1) == Approx(0));
    CHECK(value(3, 2) == Approx(0));
    value(1, 1) = input(1, 1);
    value(3, 2) = input(3, 2);
    CHECK(value.isApprox(input));
  }

  SECTION("Complex matrix") {
    t_cMatrix input = t_cMatrix::Ones(5, 5) + t_cMatrix::Random(5, 5) * 0.55;
    input.real()(1, 1) *= -1; input.real()(3, 2) *= -1;

    auto const expr = positive_quadrant(input);
    CAPTURE(input);
    CAPTURE(expr);
    CHECK(expr.imag().isApprox(t_rMatrix::Zero(5, 5)));

    auto value = expr.eval();
    CHECK(value.real()(1, 1) == Approx(0));
    CHECK(value.real()(3, 2) == Approx(0));
    value(1, 1) = input.real()(1, 1);
    value(3, 2) = input.real()(3, 2);
    CHECK(value.real().isApprox(input.real()));
    CHECK(value.imag().isApprox(0e0 * input.real()));
  }
}
