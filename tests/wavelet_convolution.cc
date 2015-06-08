#include <random>
#include "catch.hpp"

#include "wavelets/wavelets.h"
#include "wavelets/convolve.impl.cc"

typedef Eigen::Matrix<sopt::t_int, Eigen::Dynamic, 1> t_iVector;
t_iVector even(t_iVector const & x) {
  t_iVector result((x.size()+1) >> 1);
  for(t_iVector::Index i(0); i < x.size(); i += 2)
    result(i >> 1) = x(i);
  return result;
};
t_iVector odd(t_iVector const & x) {
  t_iVector result(x.size() >> 1);
  for(t_iVector::Index i(1); i < x.size(); i += 2)
    result(i >> 1) = x(i);
  return result;
};
t_iVector upsample(t_iVector const & input) {
  t_iVector result(input.size() * 2);
  for(t_iVector::Index i(0); i < input.size(); ++i) {
    result(2*i) = input(i);
    result(2*i+1) = 0;
  }
  return result;
};


TEST_CASE("Wavelet transform innards with integer data", "[wavelet]") {
  using namespace sopt::wavelets;

  std::random_device rd;
  std::default_random_engine rengine(rd());
  auto random_integer = [&rd, &rengine](sopt::t_int min, sopt::t_int max) {
    std::uniform_int_distribution<sopt::t_int> uniform_dist(min, max);
    return uniform_dist(rengine);
  };
  auto random_ivector = [&rd, &rengine](sopt::t_int size, sopt::t_int min, sopt::t_int max) {
    t_iVector result(size);
    std::uniform_int_distribution<sopt::t_int> uniform_dist(min, max);
    for(t_iVector::Index i(0); i < result.size(); ++i)
      result(i) = uniform_dist(rengine);
    return result;
  };

  t_iVector small(3); small << 1, 2, 3;
  t_iVector large(6); large << 4, 5, 6, 7, 8, 9;

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
    t_iVector result(large.size());

    convolve(result, large, small);

    CHECK(result(0) == 1 * 4 + 2 * 5 + 3 * 6);
    CHECK(result(1) == 1 * 5 + 2 * 6 + 3 * 7);
    CHECK(result(3) == 1 * 7 + 2 * 8 + 3 * 9);
    CHECK(result(4) == 1 * 8 + 2 * 9 + 3 * 4);
  }

  SECTION("Convolve and sum") {
    t_iVector result(large.size());
    t_iVector noOffset(large.size());

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
      t_iVector result(large.size());
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

  SECTION("Convolve and Down-sample simultaneously") {
    t_iVector expected(large.size());
    convolve(expected, large, small);
    t_iVector actual(large.size() >> 1);
    down_convolve(actual, large, small);
    for(size_t i(0); i < static_cast<size_t>(actual.size()); ++i)
      CHECK(expected(i << 1) == actual(i));
  }

  SECTION("Convolve output to expression") {
    t_iVector actual(large.size() << 1);
    t_iVector expected(large.size());
    convolve(std::move(actual.head(large.size())), large, small);
    convolve(expected, large, small);
    CHECK(actual.head(large.size()) == expected);
  }

  SECTION("Copy does copy") {
    auto result = copy(large);
    CHECK(large.data() != result.data());

    auto actual = copy(large.head(3));
    CHECK(large.data() != actual.data());
    CHECK(large.data() == large.head(3).data());
  }

  SECTION("Convolve, Sum and Up-sample simultaneously") {
    for(sopt::t_int i(0); i < 100; ++i) {
      auto const Ncoeffs = random_integer(2, 100) * 2;
      auto const Nfilters = random_integer(2, 100);
      auto const Nhead = Ncoeffs / 2;
      auto const Ntail = Ncoeffs - Nhead;

      auto const coeffs = random_ivector(Ncoeffs, -10, 10);
      auto const low = random_ivector(Nfilters, -10, 10);
      auto const high = random_ivector(Nfilters, -10, 10);

      t_iVector actual(Ncoeffs), expected(Ncoeffs);
      // does all in go, more complicated but compuationally less intensive
      up_convolve_sum(actual, coeffs, even(low), odd(low), even(high), odd(high));
      // first up-samples, then does convolve: conceptually simpler but does unnecessary operations
      convolve_sum(expected, upsample(coeffs.head(Nhead)), low, upsample(coeffs.tail(Ntail)), high);
      CHECK(actual.transpose() == expected.transpose());
    }
  }
}

TEST_CASE("Wavelet transform with floating point data", "[wavelet]") {
  using namespace sopt;
  using namespace sopt::wavelets;

  t_rMatrix const data = t_rMatrix::Random(6, 6);
  auto const &wavelet = Daubechies2;
  SECTION("Direct one dimensional transform == two downsample + convolution") {
     auto const actual = transform(data.row(0).transpose(), 1, wavelet).eval();
     t_rVector high(data.cols() >> 1), low(data.cols() >> 1);
     down_convolve(high, data.row(0).transpose(), wavelet.direct_filter.high);
     down_convolve(low, data.row(0).transpose(), wavelet.direct_filter.low);
     CHECK(low.isApprox(actual.head(data.row(0).size() >> 1)));
     CHECK(high.isApprox(actual.tail(data.row(0).size() >> 1)));
  }
}
