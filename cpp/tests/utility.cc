#include <random>
#include <utility>
#include <numeric>
#include "catch.hpp"

#include "sopt/utility.h"
#include "sopt/sampling.h"
#include "sopt/relative_variation.h"
#include "sopt/types.h"

TEST_CASE("Projector on positive quadrant", "[utility][project]") {
  using namespace sopt;

  SECTION("Real matrix") {
    Image<> input = Image<>::Ones(5, 5) + Image<>::Random(5, 5) * 0.55;
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
    Image<t_complex> input = Image<t_complex>::Ones(5, 5) + Image<t_complex>::Random(5, 5) * 0.55;
    input.real()(1, 1) *= -1; input.real()(3, 2) *= -1;

    auto const expr = positive_quadrant(input);
    CAPTURE(input);
    CAPTURE(expr);
    CHECK(expr.imag().isApprox(Image<>::Zero(5, 5)));

    auto value = expr.eval();
    CHECK(value.real()(1, 1) == Approx(0));
    CHECK(value.real()(3, 2) == Approx(0));
    value(1, 1) = input.real()(1, 1);
    value(3, 2) = input.real()(3, 2);
    CHECK(value.real().isApprox(input.real()));
    CHECK(value.imag().isApprox(0e0 * input.real()));
  }
}

TEST_CASE("Weighted l1 norm", "[utility][l1]") {
  sopt::Array<> weight(4);
  weight << 1, 2, 3, 4;

  SECTION("Real valued") {
    sopt::Array<> input(4);
    input << 5, -6, 7, -8;
    CHECK(sopt::l1_norm(input, weight) == Approx(5 + 12 + 21 + 32));
  }
  SECTION("Complex valued") {
    sopt::t_complex const i(0, 1);
    sopt::Array<sopt::t_complex> input(4);
    input << 5. + 5.*i, 6. + 6.*i, 7. + 7.*i, 8. + 8.*i;
    CHECK(sopt::l1_norm(input, weight) == Approx(std::sqrt(2) * (5 + 12 + 21 + 32)));
  }
}

TEST_CASE("Soft threshhold", "[utility][threshhold]") {
  sopt::Array<> input(6);
  input << 1e1, 2e1, 3e1, 4e1, 1e4, 2e4;

  SECTION("Single-valued threshhold") {
    // check thresshold
    CHECK(sopt::soft_threshhold(input, 1.1e1)(0) == Approx(0));
    CHECK(not (sopt::soft_threshhold(input, 1.1e1)(1) == Approx(0)));

    // check linearity
    auto a = sopt::soft_threshhold(input, 9e0)(0);
    auto b = sopt::soft_threshhold(input, 4.5e0)(0);
    auto c = sopt::soft_threshhold(input, 2.25e0)(0);
    CAPTURE(b - a);
    CAPTURE(c - b);
    CHECK((b - a) == Approx(2*(c - b)));
  }

  SECTION("Multi-values threshhold") {
    sopt::Array<> threshhold(6);
    input[2] *= -1;
    threshhold << 1.1e1, 1.1e1, 1e0, 4.5, 2.25;

    sopt::Array<> const actual = sopt::soft_threshhold(input, threshhold);
    CHECK(actual(0) == 0e0);
    CHECK(actual(1) == input(1) - threshhold(1));
    CHECK(actual(2) == input(2) + threshhold(2));
    CHECK(actual(3) == input(3) - threshhold(3));
  }
}

TEST_CASE("Sampling", "[utility][sampling]") {
  typedef sopt::Vector<int> t_Vector;
  t_Vector const input = t_Vector::Random(12);

  sopt::Sampling const sampling(12, {1, 3, 6, 5});

  t_Vector down = t_Vector::Zero(4);
  sampling(down, input);
  CHECK(down.size() == 4);
  CHECK(down(0) == input(1)); CHECK(down(1) == input(3));
  CHECK(down(2) == input(6)); CHECK(down(3) == input(5));

  t_Vector up = t_Vector::Zero(input.size());
  sampling.adjoint(up, down);
  CHECK(up(1) == input(1)); CHECK(up(3) == input(3));
  CHECK(up(6) == input(6)); CHECK(up(5) == input(5));
  up(1) = 0; up(3) = 0; up(6) = 0; up(5) = 0;
  CHECK(up == t_Vector::Zero(up.size()));
}

TEST_CASE("Relative variation", "[utility][convergence]") {
  sopt::RelativeVariation<double> relvar(1e-8);

  sopt::Array<> input = sopt::Array<>::Random(6);
  CHECK(not relvar(input));
  CHECK(relvar(input));
  CHECK(relvar(input + relvar.epsilon() * 0.5/6. * sopt::Array<>::Random(6)));
  CHECK(not relvar(input + relvar.epsilon() * 1.1 * sopt::Array<>::Ones(6)));
}

// Checks type traits work
static_assert(not sopt::details::HasValueType<double>::value, "");
static_assert(not sopt::details::HasValueType<std::pair<double, int>>::value, "");
static_assert(sopt::details::HasValueType<std::complex<double>>::value, "");
static_assert(sopt::details::HasValueType<sopt::Image<sopt::t_complex>::Scalar>::value, "");

static_assert(
    std::is_same<sopt::real_type<sopt::t_real>::type, sopt::t_real>::value, "");
static_assert(
    std::is_same<sopt::real_type<sopt::t_complex>::type, sopt::t_real>::value, "");
