#include <random>
#include "catch.hpp"

#include "wavelets/sara.h"

std::random_device rd;
std::default_random_engine rengine(rd());
sopt::t_int random_integer(sopt::t_int min, sopt::t_int max) {
  std::uniform_int_distribution<sopt::t_int> uniform_dist(min, max);
  return uniform_dist(rengine);
};

TEST_CASE("Check SARA implementation mechanically", "[wavelet]") {
  using namespace sopt::wavelets;
  using namespace sopt;

  typedef std::tuple<std::string, sopt::t_uint> t_i;
  SARA const sara{t_i{"DB3", 1}, t_i{"DB1", 2}, t_i{"DB1", 3}};
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

  t_rMatrix input = t_rMatrix::Random((1u << sara.max_levels()) * 3, (1u << sara.max_levels()));
  t_rMatrix coeffs;
  sara.direct(coeffs, input);

  SECTION("Direct transform") {
    t_rMatrix const first = sara[0].direct(input) / std::sqrt(sara.size());
    t_rMatrix const second = sara[1].direct(input) / std::sqrt(sara.size());
    t_rMatrix const third = sara[2].direct(input) / std::sqrt(sara.size());

    auto const N = input.cols();
    CAPTURE(coeffs.leftCols(N));
    CAPTURE(first);
    CHECK(coeffs.leftCols(N).isApprox(first));
    CHECK(coeffs.leftCols(2*N).rightCols(N).isApprox(second));
    CHECK(coeffs.rightCols(N).isApprox(third));
  }

  SECTION("Indirect transform") {
    auto const output = sara.indirect(coeffs);
    CHECK(output.isApprox(input));
  }
}
