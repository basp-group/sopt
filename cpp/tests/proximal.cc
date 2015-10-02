#include <random>
#include <utility>
#include <numeric>
#include "catch.hpp"

#include "sopt/proximal.h"

TEST_CASE("L2Ball", "[proximal]") {
  using namespace sopt;
  proximal::L2Ball<t_real> ball(0.5);
  Vector<t_real> out(5);
  Vector<t_real> x(5); x << 1, 2, 3, 4, 5;

  ball(out, x);
  CHECK(x.isApprox(out / 0.5 * x.stableNorm()));
  ball.epsilon(x.stableNorm() * 1.001);
  ball(out, 0, x); // using proximal signature call this time around
  CHECK(x.isApprox(out));
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
  translated(out, 0, x);
  Vector<t_real> expected(5);
  ball(expected, 1, x * 0.5); expected += x * 0.5;
  CHECK(out.isApprox(expected));
}
