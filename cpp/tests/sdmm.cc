#include <catch.hpp>
#include <random>

#include <Eigen/Dense>

#include "sopt/proximal.h"
#include "sopt/sdmm.h"
#include "sopt/types.h"

sopt::t_int random_integer(sopt::t_int min, sopt::t_int max) {
  extern std::unique_ptr<std::mt19937_64> mersenne;
  std::uniform_int_distribution<sopt::t_int> uniform_dist(min, max);
  return uniform_dist(*mersenne);
};

typedef sopt::t_real Scalar;
typedef sopt::Vector<Scalar> t_Vector;
typedef sopt::Matrix<Scalar> t_Matrix;

auto const N = 4;

// Makes members public so we can test one at a time
class IntrospectSDMM : public sopt::algorithm::SDMM<Scalar> {
public:
  using sopt::algorithm::SDMM<Scalar>::initialization;
  using sopt::algorithm::SDMM<Scalar>::solve_for_xn;
  using sopt::algorithm::SDMM<Scalar>::update_directions;
  using sopt::algorithm::SDMM<Scalar>::t_Vectors;
  using sopt::algorithm::SDMM<Scalar>::t_Vector;
};

TEST_CASE("Proximal translation", "[proximal]") {
  using namespace sopt;
  t_Vector const translation = t_Vector::Ones(N) * 5;
  auto const g = proximal::EuclidianNorm();
  auto const gT = proximal::translate(g, -translation);
  t_Vector const input = t_Vector::Random(N).array() + 1e0;
  CHECK(g(0.1, input).isApprox((1e0 - 0.1 / input.stableNorm()) * input));
  auto const gamma = input.stableNorm() * 0.5;
  CHECK(g(gamma, input).isApprox((1e0 - gamma / input.stableNorm()) * input));
  CHECK(g(gamma * 2 + 1, input).isApprox(input.Zero(N)));
  CHECK(gT(0.1, input)
            .isApprox((1e0 - 0.1 / (input - translation).stableNorm()) * (input - translation)
                      + translation));
}

// Iterate through algorithm for special case where the L_i are identies and the objective functions
// are simple euclidian norms
TEST_CASE("Introspect SDMM with L_i = Identity and Euclidian objectives", "[sdmm]") {
  using namespace sopt;

  t_Matrix const Id = t_Matrix::Identity(N, N).eval();
  t_Vector const target0 = t_Vector::Zero(N);
  t_Vector const target1 = t_Vector::Random(N);

  auto const g0 = proximal::translate(proximal::EuclidianNorm(), -target0);
  auto const g1 = proximal::translate(proximal::EuclidianNorm(), -target1);
  t_Vector const input = 10 * t_Vector::Random(N);

  IntrospectSDMM sdmm = IntrospectSDMM();
  sdmm.itermax(10)
      .gamma(0.01)
      .conjugate_gradient(std::numeric_limits<t_uint>::max(), 1e-12)
      .append(g0, Id)
      .append(g1, Id);

  SECTION("Step by Step") {
    INFO("Initialization");
    t_Vector out = input;
    IntrospectSDMM::t_Vectors y(sdmm.transforms().size(), t_Vector::Zero(out.size()));
    IntrospectSDMM::t_Vectors z(sdmm.transforms().size(), t_Vector::Zero(out.size()));
    sdmm.initialization(y, z, out);
    CHECK(y[0].isApprox(input));
    CHECK(y[1].isApprox(input));

    INFO("\nThen solve for conjugate gradient");
    auto const diagnostic0 = sdmm.solve_for_xn(out, y, z);
    CHECK(diagnostic0.good);
    CAPTURE(out.transpose());
    CAPTURE(input.transpose());
    CAPTURE(0.5 * (y[0] + y[1]).transpose());
    CHECK(out.isApprox(0.5 * (y[0] + y[1]), 1e-8));
    CHECK(out.isApprox(input, 1e-8));

    INFO("\nWe move on to first iteration!");
    INFO("- updates y and z");
    sdmm.update_directions(y, z, out);
    CHECK(y[0].isApprox(g0(sdmm.gamma(), input)));
    CHECK(y[1].isApprox(g1(sdmm.gamma(), input)));
    CHECK(z[0].isApprox(input - y[0]));
    CHECK(z[1].isApprox(input - y[1]));

    INFO("- solve for conjugate gradient");
    auto const diagnostic1 = sdmm.solve_for_xn(out, y, z);
    CHECK(diagnostic1.good);
    CAPTURE(out.transpose());
    CAPTURE((0.5 * (y[0] - z[0] + y[1] - z[1])).transpose());
    CHECK(out.isApprox(0.5 * (y[0] - z[0] + y[1] - z[1])));
    t_Vector const x1 = g0(sdmm.gamma(), input) + g1(sdmm.gamma(), input) - input;
    CHECK(out.isApprox(x1));

    INFO("\nWe move on to second iteration!");
    INFO("- updates y and z");
    sdmm.update_directions(y, z, out);
    CHECK(y[0].isApprox(g0(sdmm.gamma(), g1(sdmm.gamma(), input))));
    CHECK(y[1].isApprox(g1(sdmm.gamma(), g0(sdmm.gamma(), input))));
    CHECK(z[0].isApprox(g1(sdmm.gamma(), input) - y[0]));
    CHECK(z[1].isApprox(g0(sdmm.gamma(), input) - y[1]));

    INFO("- solve for conjugate gradient");
    auto const diagnostic2 = sdmm.solve_for_xn(out, y, z);
    CHECK(diagnostic2.good);
    CHECK(out.isApprox(0.5 * (y[0] - z[0] + y[1] - z[1])));
    t_Vector const x2 = g0(sdmm.gamma(), g1(sdmm.gamma(), input))
                        + g1(sdmm.gamma(), g0(sdmm.gamma(), input)) - 0.5 * g1(sdmm.gamma(), input)
                        - 0.5 * g0(sdmm.gamma(), input);
    CHECK(out.isApprox(x2));
  }

  SECTION("Iteration by Iteration") {
    t_Vector out;
    SECTION("First Iteration") {
      sdmm.itermax(1);
      auto const diagnostic = sdmm(out, input);
      CHECK(not diagnostic.good);
      CHECK(diagnostic.niters == 1);
      CHECK(out.isApprox(g0(sdmm.gamma(), input) + g1(sdmm.gamma(), input) - input));
    }
    SECTION("Second Iteration") {
      sdmm.itermax(2);
      auto const diagnostic = sdmm(out, input);
      CHECK(not diagnostic.good);
      CHECK(diagnostic.niters == 2);
      t_Vector const x2 = g0(sdmm.gamma(), g1(sdmm.gamma(), input))
                          + g1(sdmm.gamma(), g0(sdmm.gamma(), input))
                          - 0.5 * g1(sdmm.gamma(), input) - 0.5 * g0(sdmm.gamma(), input);
      CHECK(out.isApprox(x2));
    }

    SECTION("Nth Iterations") {
      sdmm.gamma(1);
      for(t_uint itermax(0); itermax < 10; ++itermax) {
        t_Vector x = input;
        t_Vector y[2] = {x, x};
        t_Vector z[2] = {t_Vector::Zero(N).eval(), t_Vector::Zero(N).eval()};
        for(t_uint i(0); i < itermax; ++i) {
          y[0] = g0(sdmm.gamma(), x + z[0]);
          y[1] = g1(sdmm.gamma(), x + z[1]);
          z[0] += x - g0(sdmm.gamma(), x + z[0]);
          z[1] += x - g1(sdmm.gamma(), x + z[1]);
          x = 0.5 * (y[0] - z[0] + y[1] - z[1]);
        }

        sdmm.itermax(itermax);
        auto const diagnostic = sdmm(out, input);
        CHECK(out.isApprox(x, 1e-8));
        CHECK(not diagnostic.good);
        CHECK(diagnostic.niters == itermax);
      }
    }
  }
}

TEST_CASE("SDMM with ||x - x0||_2 functions", "[sdmm][integration]") {
  using namespace sopt;

  t_Matrix const Id = t_Matrix::Identity(N, N).eval();
  t_Vector const target0 = t_Vector::Random(N);
  t_Vector target1 = t_Vector::Random(N) * 4;
  // for(t_uint i(0); i < N; ++i) target1(i) = i + 1;

  auto sdmm = algorithm::SDMM<Scalar>()
                  .itermax(5000)
                  .gamma(1)
                  .conjugate_gradient(std::numeric_limits<t_uint>::max(), 1e-12)
                  .append(proximal::translate(proximal::EuclidianNorm(), -target0), Id)
                  .append(proximal::translate(proximal::EuclidianNorm(), -target1), Id);

  t_Vector result;
  SECTION("Just two operators") {
    auto const diagnostic = sdmm(result, t_Vector::Random(N));
    CHECK(not diagnostic.good);
    CHECK(diagnostic.niters == sdmm.itermax());
    t_Vector const segment = (target1 - target0).normalized();
    t_real const alpha = (result - target0).transpose() * segment;
    CAPTURE(target0.transpose());
    CAPTURE(target1.transpose());
    CHECK((target1 - target0).transpose() * segment >= alpha);
    CHECK(alpha >= 0e0);
    CHECK((result - target0 - alpha * segment).stableNorm() < 1e-8);
  }

  SECTION("Three operators") {
    t_Vector const target2 = t_Vector::Random(N) * 8;
    sdmm.append(proximal::translate(proximal::EuclidianNorm(), -target2), Id);
    auto const diagnostic = sdmm(result, t_Vector::Random(N));
    CHECK(not diagnostic.good);
    CHECK(diagnostic.niters == sdmm.itermax());
    CAPTURE(result.transpose());
    auto const func = [&target0, &target1, &target2](t_Vector const &x) {
      return (x - target0).stableNorm() + (x - target1).stableNorm() + (x - target2).stableNorm();
    };
    for(int i(0); i < N; ++i) {
      t_Vector epsilon = t_Vector::Zero(N);
      epsilon(i) = 1e-6;
      CHECK(func(result) < func(result + epsilon));
      CHECK(func(result) < func(result - epsilon));
    }
  }

  SECTION("With different L") {
    t_Matrix const L0 = t_Matrix::Random(N, N) * 2;
    t_Matrix const L1 = t_Matrix::Random(N, N) * 4;
    REQUIRE(std::abs((L0.transpose() * L0 + L1.transpose() * L1).determinant()) > 1e-8);
    sdmm.itermax(300);
    sdmm.transforms(0) = linear_transform(L0);
    sdmm.transforms(1) = linear_transform(L1);
    auto const diagnostic = sdmm(result, t_Vector::Random(N));
    CHECK(not diagnostic.good);
    CHECK(diagnostic.niters == sdmm.itermax());
    CAPTURE(result.transpose());
    auto const func = [&target0, &target1, &L0, &L1](t_Vector const &x) {
      return (L0 * x - target0).stableNorm() + (L1 * x - target1).stableNorm();
    };
    for(int i(0); i < N; ++i) {
      t_Vector epsilon = t_Vector::Zero(N);
      epsilon(i) = 1e-4;
      CAPTURE(epsilon.transpose());
      CHECK(func(result) <= func(result + epsilon));
      CHECK(func(result) <= func(result - epsilon));
    }
  }
}
