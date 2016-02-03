#include <random>
#include "catch.hpp"

#include "sopt/wrapper.h"

TEST_CASE("Function wrappers", "[utility]") {
  using namespace sopt;
  typedef Array<int> t_Array;
  typedef t_Array &t_RefArray;
  typedef t_Array const t_ConstRefArray;

  SECTION("Square function") {
    auto func = [](t_RefArray output, t_ConstRefArray const &input) { output = input * 2 + 1; };

    t_Array const x = t_Array::Random(5);
    auto const A = details::wrap<t_Array>(func);
    // Expected result
    t_Array const expected = (x * 2 + 1).eval();

    CHECK((A * x).matrix() == expected.matrix());
    CHECK(A(x).matrix() == expected.matrix());
  }

  SECTION("Rectangular function") {
    auto func = [](t_RefArray output, t_ConstRefArray const &input) {
      output = input.head(input.size() / 2) * 2 + 1;
    };

    t_Array const x = t_Array::Random(5);
    auto const A = details::wrap<t_Array>(func, {{1, 2, 0}});
    // Expected result
    t_Array const expected = (x.head(x.size() / 2) * 2 + 1).eval();

    CHECK((A * x).cols() == 1);
    CHECK((A * x).rows() == 2);
    CHECK((A * x).matrix() == expected.matrix());
    CHECK(A(x).matrix() == expected.matrix());
  }

  SECTION("Fixed output-size functions") {
    auto func
        = [](t_RefArray output, t_ConstRefArray const &input) { output = input.head(3) * 2 + 1; };

    t_Array const x = t_Array::Random(5);
    auto const A = details::wrap<t_Array>(func, {{0, 1, 3}});
    // Expected result
    t_Array const expected = (x.head(3) * 2 + 1).eval();

    CHECK((A * x).cols() == 1);
    CHECK((A * x).rows() == 3);
    CHECK((A * x).matrix() == expected.matrix());
    CHECK(A(x).matrix() == expected.matrix());
  }
}
