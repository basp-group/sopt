#include "catch.hpp"

#include "wavelets/convolve.impl.cc"

TEST_CASE("Periodic convolution operator", "[wavelet]") {
  using namespace sopt::wavelets;

  Eigen::Matrix<int, Eigen::Dynamic, 1> small(3); small << 1, 2, 3;
  Eigen::Matrix<int, Eigen::Dynamic, 1> large(6); large << 4, 5, 6, 7, 8, 9;

  SECTION("Periodic scalar product") {

    // no wrapping
    CHECK(periodic_scalar_product(large, small, 0) == 1 * 4 + 2 * 5 + 3 * 6);
    CHECK(periodic_scalar_product(large, small, 1) == 1 * 5 + 2 * 6 + 3 * 7);
    CHECK(periodic_scalar_product(large, small, 3) == 1 * 7 + 2 * 8 + 3 * 9);

    // with wrapping
    CHECK(periodic_scalar_product(large, small, 4) == 1 * 8 + 2 * 9 + 3 * 4);
    // with wrapping and expression
    CHECK(periodic_scalar_product(large, small.reverse(), 4) == 3 * 8 + 2 * 9 + 1 * 4);

    // signal smaller than filter
    CHECK(periodic_scalar_product(small, large.head(4), 1) == 4 * 2 + 5 * 3 + 6 * 1 + 7 * 2);
  }

  SECTION("Convolve") {
    Eigen::Matrix<int, Eigen::Dynamic, 1> result(large.size());

    convolve(result, large, small);

    CHECK(result(0) == 3 * 4 + 2 * 5 + 1 * 6);
    CHECK(result(1) == 3 * 5 + 2 * 6 + 1 * 7);
    CHECK(result(3) == 3 * 7 + 2 * 8 + 1 * 9);
    CHECK(result(4) == 3 * 8 + 2 * 9 + 1 * 4);
  }
}
