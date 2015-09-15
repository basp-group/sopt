#include <random>
#include "catch.hpp"

#include "wavelets/wavelets.h"
#include "wavelets/wavelet_data.h"
#include "wavelets/indirect.h"
#include "wavelets/direct.h"

typedef Eigen::Array<sopt::t_int, Eigen::Dynamic, 1> t_iVector;
t_iVector even(t_iVector const & x) {
  t_iVector result((x.size()+1) / 2);
  for(t_iVector::Index i(0); i < x.size(); i += 2)
    result(i / 2) = x(i);
  return result;
};
t_iVector odd(t_iVector const & x) {
  t_iVector result(x.size() / 2);
  for(t_iVector::Index i(1); i < x.size(); i += 2)
    result(i / 2) = x(i);
  return result;
};
template<class T>
Eigen::Array<typename T::Scalar, T::RowsAtCompileTime, T::ColsAtCompileTime>
upsample(Eigen::ArrayBase<T> const & input) {
  typedef Eigen::Array<typename T::Scalar, T::RowsAtCompileTime, T::ColsAtCompileTime> Matrix;
  Matrix result(input.size() * 2);
  for(t_iVector::Index i(0); i < input.size(); ++i) {
    result(2*i) = input(i);
    result(2*i+1) = 0;
  }
  return result;
};

std::random_device rd;
std::default_random_engine rengine(rd());
sopt::t_int random_integer(sopt::t_int min, sopt::t_int max) {
  std::uniform_int_distribution<sopt::t_int> uniform_dist(min, max);
  return uniform_dist(rengine);
};
t_iVector random_ivector(sopt::t_int size, sopt::t_int min, sopt::t_int max) {
  t_iVector result(size);
  std::uniform_int_distribution<sopt::t_int> uniform_dist(min, max);
  for(t_iVector::Index i(0); i < result.size(); ++i)
    result(i) = uniform_dist(rengine);
  return result;
};

// Checks round trip operation
template<class T0>
  void check_round_trip(
      Eigen::ArrayBase<T0> const& input_, sopt::t_uint db, sopt::t_uint nlevels=1) {
    auto const input = input_.eval();
    auto const &dbwave = sopt::wavelets::daubechies_data(db);
    auto const transform = sopt::wavelets::direct_transform(input, nlevels, dbwave);
    auto const actual = sopt::wavelets::indirect_transform(transform, nlevels, dbwave);
    CAPTURE(actual);
    CAPTURE(input);
    CAPTURE(transform);
    CHECK(input.isApprox(actual, 1e-14));
    CHECK(not transform.isApprox(sopt::wavelets::direct_transform(input, nlevels-1, dbwave), 1e-4));
  }


TEST_CASE("Wavelet transform innards with integer data", "[wavelet]") {
  using namespace sopt::wavelets;

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
    CHECK((trial(0, 1, 3, 1) == trial(0, 1, 1, 3)).all());
    CHECK((trial(5, 1, 3, 1) == trial(3, 1, 5, 1)).all());
    CHECK((trial(1, 5, 3, 1) == trial(3, 1, 5, 1)).all());
    CHECK((trial(1, 3, 5, 1) == trial(3, 1, 5, 1)).all());
    CHECK((trial(1, 3, 1, 5) == trial(3, 1, 5, 1)).all());
    CHECK((trial(1, 0, 4, 2) == trial(3, 1, 5, 1)).all());
    CHECK((trial(1, -1, 1, 1) == trial(0, 1, 0, 1)).all());
    CHECK((trial(4, -3, 2, 6) == trial(0, 1, 0, 1)).all());
  }

  SECTION("Convolve and Down-sample simultaneously") {
    t_iVector expected(large.size());
    convolve(expected, large, small);
    t_iVector actual(large.size() / 2);
    down_convolve(actual, large, small);
    for(size_t i(0); i < static_cast<size_t>(actual.size()); ++i)
      CHECK(expected(i * 2) == actual(i));
  }

  SECTION("Convolve output to expression") {
    t_iVector actual(large.size() * 2);
    t_iVector expected(large.size());
    convolve(std::move(actual.head(large.size())), large, small);
    convolve(expected, large, small);
    CHECK((actual.head(large.size()) == expected).all());
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
      auto const Ncoeffs = random_integer(2, 10) * 2;
      auto const Nfilters = random_integer(2, 5);
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
      CHECK((actual == expected).all());
    }
  }
}

TEST_CASE("1D wavelet transform with floating point data", "[wavelet]") {
  using namespace sopt;
  using namespace sopt::wavelets;

  t_rMatrix const data = t_rMatrix::Random(16, 16);
  auto const &wavelet = daubechies_data(4);

  // Condition on input fixture data
  REQUIRE((data.rows() %  2 == 0 and (data.cols() == 1 or data.cols() % 2 == 0)));

  SECTION("Direct transform == two downsample + convolution") {
     auto const actual = direct_transform(data.row(0), 1, wavelet);
     t_rVector high(data.cols() / 2), low(data.cols() / 2);
     down_convolve(high, data.row(0), wavelet.direct_filter.high);
     down_convolve(low, data.row(0), wavelet.direct_filter.low);
     CHECK(low.transpose().isApprox(actual.head(data.row(0).size() / 2)));
     CHECK(high.transpose().isApprox(actual.tail(data.row(0).size() / 2)));
  }

  SECTION("Indirect transform == two upsample + convolution") {
     auto const actual = indirect_transform(data.row(0).transpose(), 1, wavelet);
     auto const low = upsample(data.row(0).transpose().head(data.rows() / 2));
     auto const high = upsample(data.row(0).transpose().tail(data.rows() / 2));
     auto expected = copy(data.row(0).transpose());
     convolve_sum(
         expected,
         low, wavelet.direct_filter.low.reverse(),
         high, wavelet.direct_filter.high.reverse()
     );
     CAPTURE(expected.transpose());
     CAPTURE(actual.transpose());
     CHECK(expected.isApprox(actual));
  }

  SECTION("Round-trip test for single level") {
    for(t_int i(0); i < 20; ++i) {
      check_round_trip(t_rVector::Random(random_integer(2, 100)*2), random_integer(1, 38), 1);
    }
  }

  SECTION("Round-trip test for two levels") {
    check_round_trip(t_rVector::Random(8), 1, 2);
    check_round_trip(t_rVector::Random(8), 2, 2);
    check_round_trip(t_rVector::Random(16), 4, 2);
    check_round_trip(t_rVector::Random(52), 10, 2);
  }

  t_uint nlevels = 5;
  SECTION("Round-trip test for multiple levels") {
    for(t_int i(0); i < 10; ++i) {
      auto const n = random_integer(2, nlevels);
      check_round_trip(
        t_rVector::Random(random_integer(2, 100) * (1u << n)),
        random_integer(1, 38),
        n
      );
    }
  }
}

TEST_CASE("1D wavelet transform with complex data", "[wavelet]") {
  using namespace sopt;
  using namespace sopt::wavelets;
  SECTION("Round-trip test for complex data") {
    auto input = t_cVector::Random(random_integer(2, 100)*2).eval();
    auto const &dbwave = daubechies_data(random_integer(1, 38));
    auto const actual = indirect_transform(direct_transform(input, 1, dbwave), 1, dbwave);
    CHECK(input.isApprox(actual, 1e-14));
    CHECK(not input.isApprox(direct_transform(input, 1, dbwave), 1e-4));
  }
}

TEST_CASE("2D wavelet transform with real data", "[wavelet]") {
  using namespace sopt;
  using namespace sopt::wavelets;
  SECTION("Single level round-trip test for square matrix") {
    auto N = random_integer(2, 100) * 2;
    check_round_trip(t_rMatrix::Random(N, N), random_integer(1, 38), 1);
  }
  SECTION("Single level round-trip test for non-square matrix") {
    auto Nx = random_integer(2, 5) * 2;
    auto Ny = Nx + 5 * 2;
    check_round_trip(t_rMatrix::Random(Nx, Ny), random_integer(1, 38), 1);
  }
  SECTION("Round-trip test for multiple levels") {
    for(t_int i(0); i < 10; ++i) {
      auto const n = random_integer(2, 5);
      auto const Nx = random_integer(2, 5) * (1u << n);
      auto const Ny = random_integer(2, 5) * (1u << n);
      check_round_trip(t_rMatrix::Random(Nx, Ny), random_integer(1, 38), n);
    }
  }
}

TEST_CASE("Functor implementation", "[wavelet]") {
  using namespace sopt;
  auto const wavelet = wavelets::factory("DB3", 4);
  auto const input = t_cMatrix::Random(256, 128).eval();
  SECTION("Normal instances") {
    auto const transform = wavelet.direct(input);
    CHECK(transform.isApprox(wavelets::direct_transform(input, wavelet.levels(), wavelet)));
    CHECK(input.isApprox(wavelet.indirect(transform)));
  }
  SECTION("Expression instances") {
    t_cMatrix output(2, input.cols());
    wavelet.direct(output.row(0).transpose(), input.row(0).transpose());
    wavelet.indirect(output.row(0).transpose(), output.row(1).transpose());
    CHECK(input.row(0).isApprox(output.row(1)));
  }
}

TEST_CASE("Automatic input resizing", "[wavelet]") {
  using namespace sopt;
  auto const wavelet = wavelets::factory("DB3", 4);
  auto const input = t_cMatrix::Random(256, 128).eval();
  t_cMatrix output(1, 1);
  wavelet.direct(output, input);
  CHECK(output.rows() == input.rows());
  CHECK(output.cols() == input.cols());

  output.resize(1, 1);
  wavelet.indirect(input, output);
  CHECK(output.rows() == input.rows());
  CHECK(output.cols() == input.cols());
}
