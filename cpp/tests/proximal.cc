#include <random>
#include <utility>
#include <numeric>
#include "catch.hpp"

#include "sopt/proximal.h"
#include "sopt/l1_proximal.h"

TEST_CASE("L2Ball", "[proximal]") {
  using namespace sopt;
  proximal::L2Ball<t_real> ball(0.5);
  Vector<t_real> out;
  Vector<t_real> x(5); x << 1, 2, 3, 4, 5;

  out = ball(0, x);
  CHECK(x.isApprox(out / 0.5 * x.stableNorm()));
  ball.epsilon(x.stableNorm() * 1.001);
  out = ball(0, x);
  CHECK(x.isApprox(out));
}

TEST_CASE("Euclidian norm", "[proximal]") {
  using namespace sopt;
  proximal::EuclidianNorm eucl;

  Vector<t_real> out(5);
  Vector<t_real> x(5); x << 1, 2, 3, 4, 5;
  eucl(out, x.stableNorm() * 1.001, x);
  CHECK(out.isApprox(Vector<t_real>::Zero(x.size())));

  out = eucl(0.1, x);
  CHECK(out.isApprox(x * (1e0 - 0.1/x.stableNorm())));
}

TEST_CASE("Translation", "[proximal]") {
  using namespace sopt;
  Vector<t_real> out(5);
  Vector<t_real> x(5); x << 1, 2, 3, 4, 5;
  proximal::L2Ball<t_real> ball(5000);
  // Pass in a reference, so we can modify ball.epsilon later in the test.
  auto const translated = proximal::translate(std::ref(ball), -x * 0.5);
  translated(out, 0, x);
  CHECK(out.isApprox(x));

  ball.epsilon(0.125);
  out = translated(0, x);
  Vector<t_real> expected = ball(1, x * 0.5) + x * 0.5;
  CHECK(out.isApprox(expected));
}

TEST_CASE("Tight-Frame L1 proximal", "[l1][proximal]") {
  using namespace sopt;
  auto const l1 = proximal::L1TightFrame<t_complex>();

  // Vector<t_complex> const input = Vector<t_complex>::Random(8);
  // Vector<t_complex> const actual = l1(1, input);
  // CHECK(proximal::l1_norm(1, input).isApprox(actual));
}
