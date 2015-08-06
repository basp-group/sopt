#include <random>
#include "catch.hpp"

#include "sopt/wrapper.h"

TEST_CASE("Function wrappers", "[utility]") {
  using namespace sopt;

  typedef Eigen::Array<int, Eigen::Dynamic, 1> Vector;
  auto func = [](Vector &output, Vector const &input) { output = input * 2 + 1; };

  Vector const x = Vector::Random(5);
  auto const A = details::wrap<Vector>(func);
  // Expected result
  Vector const expected = (x * 2 + 1).eval();

  CHECK((A * x).matrix() == expected.matrix());
  CHECK(A(x).matrix() == expected.matrix());
}
