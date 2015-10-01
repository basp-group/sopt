#include <random>
#include "catch.hpp"

#include "sopt/wrapper.h"

TEST_CASE("Function wrappers", "[utility]") {
  using namespace sopt;

  SECTION("Square function") {
    auto func = [](Array<int> &output, Array<int> const &input) { output = input * 2 + 1; };

    Array<int> const x = Array<int>::Random(5);
    auto const A = details::wrap<Array<int>>(func);
    // Expected result
    Array<int> const expected = (x * 2 + 1).eval();

    CHECK((A * x).matrix() == expected.matrix());
    CHECK(A(x).matrix() == expected.matrix());
  }

  SECTION("Rectangular function") {
    auto func = [](Array<int> &output, Array<int> const &input) {
      output = input.head(input.size()/2) * 2 + 1;
    };

    Array<int> const x = Array<int>::Random(5);
    auto const A = details::wrap<Array<int>>(func, {{1, 2, 0}});
    // Expected result
    Array<int> const expected = (x.head(x.size()/2) * 2 + 1).eval();

    CHECK((A * x).cols() == 1);
    CHECK((A * x).rows() == 2);
    CHECK((A * x).matrix() == expected.matrix());
    CHECK(A(x).matrix() == expected.matrix());
  }

  SECTION("Fixed output-size functions") {
    auto func = [](Array<int> &output, Array<int> const &input) {
      output = input.head(3) * 2 + 1;
    };

    Array<int> const x = Array<int>::Random(5);
    auto const A = details::wrap<Array<int>>(func, {{0, 1, 3}});
    // Expected result
    Array<int> const expected = (x.head(3) * 2 + 1).eval();

    CHECK((A * x).cols() == 1);
    CHECK((A * x).rows() == 3);
    CHECK((A * x).matrix() == expected.matrix());
    CHECK(A(x).matrix() == expected.matrix());
  }
}
