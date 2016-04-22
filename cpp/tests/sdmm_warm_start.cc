#include <catch.hpp>
#include <random>

#include <Eigen/Dense>

#include "sopt/proximal.h"
#include "sopt/sdmm.h"
#include "sopt/types.h"

typedef sopt::t_real Scalar;
typedef sopt::Vector<Scalar> t_Vector;
typedef sopt::Matrix<Scalar> t_Matrix;

auto const N = 30;
SCENARIO("SDMM with warm start", "[sdmm][integration]") {
  using namespace sopt;

  GIVEN("An SDMM instance with its input") {
    t_Matrix const Id = t_Matrix::Identity(N, N).eval();
    t_Vector const target0 = t_Vector::Random(N);
    t_Vector target1 = t_Vector::Random(N) * 4;

    auto convergence = [&target1, &target0](t_Vector const &x) -> bool {
      t_Vector const segment = (target1 - target0).normalized();
      t_real const alpha = (x - target0).transpose() * segment;
      return alpha >= 0e0 and (target1 - target0).transpose() * segment >= alpha
             and (x - target0 - alpha * segment).stableNorm() < 1e-8;
    };

    auto sdmm = algorithm::SDMM<Scalar>()
                    .is_converged(convergence)
                    .itermax(5000)
                    .gamma(1)
                    .conjugate_gradient(std::numeric_limits<t_uint>::max(), 1e-12)
                    .append(proximal::translate(proximal::EuclidianNorm(), -target0), Id)
                    .append(proximal::translate(proximal::EuclidianNorm(), -target1), Id);
    t_Vector input = t_Vector::Random(N);

    WHEN("the algorithms runs") {
      auto const full = sdmm(input);
      THEN("it converges") {
        CHECK(full.niters > 20);
        CHECK(full.good);
      }

      WHEN("It is set to stop before convergence") {
        auto const first_half = sdmm.itermax(full.niters - 5)(input);
        THEN("It is not converged") { CHECK(not first_half.good); }

        WHEN("A warm restart is attempted") {
          auto const second_half = sdmm.itermax(5000)(first_half);
          THEN("The warm restart is validated by the fast convergence") {
            CHECK(second_half.niters < 10);
          }
        }
      }
    }
  }
}
