#include <random>
#include "catch.hpp"

#include "sopt/wrapper.h"

TEST_CASE("Function wrappers", "[utility]") {
  using namespace sopt;

  SECTION("Square function") {
    typedef Eigen::Array<int, Eigen::Dynamic, 1> Vector;
    auto func = [](Vector &output, Vector const &input) { output = input * 2 + 1; };

    Vector const x = Vector::Random(5);
    auto const A = details::wrap<Vector>(func);
    // Expected result
    Vector const expected = (x * 2 + 1).eval();

    CHECK((A * x).matrix() == expected.matrix());
    CHECK(A(x).matrix() == expected.matrix());
  }

  SECTION("Rectangular function") {
    typedef Eigen::Array<int, Eigen::Dynamic, 1> Vector;
    auto func = [](Vector &output, Vector const &input) {
      output = input.head(input.size()/2) * 2 + 1;
    };

    Vector const x = Vector::Random(5);
    auto const A = details::wrap<Vector>(func, {{1, 2, 0}});
    // Expected result
    Vector const expected = (x.head(x.size()/2) * 2 + 1).eval();

    CHECK((A * x).cols() == 1);
    CHECK((A * x).rows() == 2);
    CHECK((A * x).matrix() == expected.matrix());
    CHECK(A(x).matrix() == expected.matrix());
  }

  SECTION("Fixed output-size functions") {
    typedef Eigen::Array<int, Eigen::Dynamic, 1> Vector;
    auto func = [](Vector &output, Vector const &input) {
      output = input.head(3) * 2 + 1;
    };

    Vector const x = Vector::Random(5);
    auto const A = details::wrap<Vector>(func, {{0, 1, 3}});
    // Expected result
    Vector const expected = (x.head(3) * 2 + 1).eval();

    CHECK((A * x).cols() == 1);
    CHECK((A * x).rows() == 3);
    CHECK((A * x).matrix() == expected.matrix());
    CHECK(A(x).matrix() == expected.matrix());
  }
}
