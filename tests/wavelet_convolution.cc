#include <random>
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
    // wrapping works with offset as well
    CHECK(periodic_scalar_product(large, small, 4 + large.size()) == 1 * 8 + 2 * 9 + 3 * 4);
    CHECK(periodic_scalar_product(large, small, 4 - 3 * large.size()) == 1 * 8 + 2 * 9 + 3 * 4);

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

  SECTION("Convolve and sum") {
    Eigen::Matrix<int, Eigen::Dynamic, 1> result(large.size());
    Eigen::Matrix<int, Eigen::Dynamic, 1> noOffset(large.size());

    // Check that if high pass is zero, then this is an offseted convolution
    convolve_sum(result, large, small, large, 0 * small);
    convolve(noOffset, large, small);
    CHECK(result(small.size() - 1) == noOffset(0));
    CHECK(result(0) == noOffset(result.size() - small.size() + 1));

    // Check same for low pass
    convolve_sum(result, large, 0 * small, large, small);
    CHECK(result(small.size() - 1) == noOffset(0));
    CHECK(result(0) == noOffset(result.size() - small.size() + 1));

    // Check symmetry relationships
    auto const trial = [&small, &large](int a, int b, int c, int d) {
      Eigen::Matrix<int, Eigen::Dynamic, 1> result(large.size());
      convolve_sum(result, a * large, b * small, c * large, d * small);
      return result;
    };

    // should all be ok as long as arguments sum: (a * b) + (c * d) == (a' * b') + (c' * d')
    CHECK(trial(0, 1, 3, 1) == trial(0, 1, 1, 3));
    CHECK(trial(5, 1, 3, 1) == trial(3, 1, 5, 1));
    CHECK(trial(1, 5, 3, 1) == trial(3, 1, 5, 1));
    CHECK(trial(1, 3, 5, 1) == trial(3, 1, 5, 1));
    CHECK(trial(1, 3, 1, 5) == trial(3, 1, 5, 1));
    CHECK(trial(1, 0, 4, 2) == trial(3, 1, 5, 1));
    CHECK(trial(1, -1, 1, 1) == trial(0, 1, 0, 1));
    CHECK(trial(4, -3, 2, 6) == trial(0, 1, 0, 1));
  }
}
