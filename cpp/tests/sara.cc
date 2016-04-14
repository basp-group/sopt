#include <catch.hpp>
#include <random>
#include <string>
#include <tuple>

#include "sopt/wavelets/sara.h"
#include "sopt/wavelets.h"

sopt::t_int random_integer(sopt::t_int min, sopt::t_int max) {
  extern std::unique_ptr<std::mt19937_64> mersenne;
  std::uniform_int_distribution<sopt::t_int> uniform_dist(min, max);
  return uniform_dist(*mersenne);
};

TEST_CASE("Check SARA implementation mechanically", "[wavelet]") {
  using namespace sopt::wavelets;
  using namespace sopt;

  typedef std::tuple<std::string, sopt::t_uint> t_i;
  SARA const sara{t_i{std::string{"DB3"}, 1u}, t_i{std::string{"DB1"}, 2u},
                  t_i{std::string{"DB1"}, 3u}};
  SECTION("Construction and vector functionality") {
    CHECK(sara.size() == 3);
    CHECK(sara[0].levels() == 1);
    CHECK(sara[1].levels() == 2);
    CHECK(sara[2].levels() == 3);
    CHECK(sara.max_levels() == 3);
    CHECK(sara[0].coefficients.isApprox(factory("DB3", 1).coefficients));
    CHECK(sara[1].coefficients.isApprox(factory("DB1", 1).coefficients));
    CHECK(sara[2].coefficients.isApprox(factory("DB1", 1).coefficients));
  }

  Image<> input = Image<>::Random((1u << sara.max_levels()) * 3, (1u << sara.max_levels()));
  Image<> coeffs;
  sara.direct(coeffs, input);

  SECTION("Direct transform") {
    Image<> const first = sara[0].direct(input) / std::sqrt(sara.size());
    Image<> const second = sara[1].direct(input) / std::sqrt(sara.size());
    Image<> const third = sara[2].direct(input) / std::sqrt(sara.size());

    auto const N = input.cols();
    CAPTURE(coeffs.leftCols(N));
    CAPTURE(first);
    CHECK(coeffs.leftCols(N).isApprox(first));
    CHECK(coeffs.leftCols(2 * N).rightCols(N).isApprox(second));
    CHECK(coeffs.rightCols(N).isApprox(third));
  }

  SECTION("Indirect transform") {
    auto const output = sara.indirect(coeffs);
    CHECK(output.isApprox(input));
  }
}

TEST_CASE("Linear-transform wrapper", "[wavelet]") {
  using namespace sopt::wavelets;
  using namespace sopt;
  SARA const sara{std::make_tuple(std::string{"DB3"}, 1u), std::make_tuple(std::string{"DB1"}, 2u),
                  std::make_tuple(std::string{"DB1"}, 3u)};

  auto const rows = 256, cols = 256;
  auto const Psi = linear_transform<t_real>(sara, rows, cols);
  SECTION("Indirect transform") {
    Image<> const image = Image<>::Random(rows, cols);
    Image<> const expected = sara.direct(image);
    // The linear transform expects a column vector as input
    auto const as_vector = Vector<>::Map(image.data(), image.size());
    // And it returns a column vector as well
    Vector<> const actual = Psi.adjoint() * as_vector;
    CHECK(actual.size() == expected.size());
    auto const coeffs = Image<>::Map(actual.data(), image.rows(), image.cols() * sara.size());
    CHECK(expected.rows() == coeffs.rows());
    CHECK(expected.cols() == coeffs.cols());
    CHECK(coeffs.isApprox(expected, 1e-8));
  }
  SECTION("direct transform") {
    Image<> const coeffs = Image<>::Random(rows, cols * sara.size());
    Image<> const expected = sara.indirect(coeffs);
    // The linear transform expects a column vector as input
    auto const as_vector = Vector<>::Map(coeffs.data(), coeffs.size());
    // And it returns a column vector as well
    Vector<> const actual = Psi * as_vector;
    CHECK(actual.size() == expected.size());
    CHECK(coeffs.cols() % sara.size() == 0);
    auto const image = Image<>::Map(actual.data(), coeffs.rows(), coeffs.cols() / sara.size());
    CHECK(expected.rows() == image.rows());
    CHECK(expected.cols() == image.cols());
    CHECK(image.isApprox(expected, 1e-8));
  }
}
