#include <catch.hpp>
#include <random>

#include <Eigen/Dense>

#include "sopt/padmm.h"
#include "sopt/proximal.h"
#include "sopt/types.h"

typedef sopt::t_real Scalar;
typedef sopt::Vector<Scalar> t_Vector;
typedef sopt::Matrix<Scalar> t_Matrix;

auto const N = 30;
SCENARIO("ProximalADMM with warm start", "[padmm][integration]") {
  using namespace sopt;

  GIVEN("A ProximalADMM instance with its input") {
    t_Vector const target0 = t_Vector::Random(N);
    t_Vector const target1 = t_Vector::Random(N) * 4;
    auto const g0 = proximal::translate(proximal::EuclidianNorm(), -target0);
    auto const g1 = proximal::translate(proximal::EuclidianNorm(), -target1);
    t_Matrix const mId = -t_Matrix::Identity(N, N);

    auto convergence = [&target1, &target0](t_Vector const &x) -> bool {
      t_Vector const segment = (target1 - target0).normalized();
      t_real const alpha = (x - target0).transpose() * segment;
      SOPT_TRACE(" {} {}", alpha, (x - target0 - alpha * segment).stableNorm());
      return alpha >= 0e0 and (target1 - target0).transpose() * segment >= alpha
             and (x - target0 - alpha * segment).stableNorm() < 1e-8;
    };

    auto padmm = algorithm::ProximalADMM<Scalar>(g0, g1, t_Vector::Zero(N))
                     .Phi(mId)
                     .itermax(3000)
                     .gamma(0.5)
                     .is_converged(convergence);

    WHEN("the algorithms runs") {
      auto const full = padmm();
      THEN("it converges") {
        CHECK(full.niters > 20);
        CHECK(full.good);
      }

      WHEN("It is set to stop before convergence") {
        auto const first_half = padmm.itermax(full.niters - 5)();
        THEN("It is not converged") { CHECK(not first_half.good); }

        WHEN("A warm restart is attempted") {
          auto const second_half = padmm.itermax(5000)(first_half);
          THEN("The warm restart is validated by the fast convergence") {
            CHECK(second_half.niters < 10);
          }
        }
      }
    }
  }
}
