#include <catch.hpp>
#include <random>

#include <Eigen/Dense>

#include "sopt/imaging_padmm.h"
#include "sopt/reweighted.h"

using namespace sopt;

//! \brief Minimum set of functions and typedefs needed by reweighting
//! \details The attributes are public and static so we can access them during the tests.
struct DummyAlgorithm {
  typedef t_real Scalar;
  typedef Vector<Scalar> t_Vector;
  typedef ConvergenceFunction<Scalar> t_IsConverged;

  struct DiagnosticAndResult {
    //! Expected by reweighted algorithm
    static t_Vector x;
  };

  DiagnosticAndResult operator()(t_Vector const &x) const {
    ++called_with_x;
    DiagnosticAndResult::x = x.array() + 0.1;
    return {};
  }
  DiagnosticAndResult operator()(DiagnosticAndResult const &warm) const {
    ++called_with_warm;
    DiagnosticAndResult::x = warm.x.array() + 0.1;
    return {};
  }

  //! Applies Ψ^T * x
  static t_Vector reweightee(DummyAlgorithm const &, t_Vector const &x) {
    ++DummyAlgorithm::called_reweightee;
    return x * 2;
  }
  //! sets the weights
  static void set_weights(DummyAlgorithm &, t_Vector const &weights) {
    ++DummyAlgorithm::called_weights;
    DummyAlgorithm::weights = weights;
  }

  static t_Vector weights;
  static int called_with_x;
  static int called_with_warm;
  static int called_reweightee;
  static int called_weights;
};

int DummyAlgorithm::called_with_x = 0;
int DummyAlgorithm::called_with_warm = 0;
int DummyAlgorithm::called_reweightee = 0;
int DummyAlgorithm::called_weights = 0;
DummyAlgorithm::t_Vector DummyAlgorithm::DiagnosticAndResult::x;
DummyAlgorithm::t_Vector DummyAlgorithm::weights;

TEST_CASE("L0-Approximation") {

  auto const N = 6;
  DummyAlgorithm::t_Vector const input = DummyAlgorithm::t_Vector::Random(N);

  auto l0algo = algorithm::reweighted(DummyAlgorithm(), DummyAlgorithm::set_weights,
                                      DummyAlgorithm::reweightee);

  DummyAlgorithm::called_with_x = 0;
  DummyAlgorithm::called_with_warm = 0;
  DummyAlgorithm::called_reweightee = 0;
  DummyAlgorithm::called_weights = 0;
  DummyAlgorithm::DiagnosticAndResult::x = DummyAlgorithm::t_Vector::Zero(0);
  DummyAlgorithm::weights = DummyAlgorithm::t_Vector::Zero(0);

  GIVEN("The maximum number of iteration is zero") {
    l0algo.itermax(0);
    WHEN("The reweighting algorithm is called") {
      auto const result = l0algo(input);
      THEN("The algorithm exited at the first iteration") {
        CHECK(result.niters == 0);
        CHECK(result.good == true);
      }
      THEN("The weights is set to 1") {
        CHECK(result.weights.size() == 1);
        CHECK(std::abs(result.weights(0) - 1) < 1e-12);
      }
      THEN("The inner algorithm was called once") {
        CHECK(DummyAlgorithm::called_with_x == 1);
        CHECK(DummyAlgorithm::called_with_warm == 0);
        CHECK(result.inner_loop.x.array().isApprox(input.array() + 0.1));
      }
    }
  }

  GIVEN("The maximum number of iterations is one") {
    l0algo.itermax(1);
    WHEN("The reweighting algorithm is called") {
      auto const result = l0algo(input);
      THEN("The algorithm exited at the second iteration") {
        CHECK(result.niters == 1);
        CHECK(result.good == true);
      }
      THEN("The weights are not one") {
        CHECK(result.weights.size() == input.size());
        // standard deviation of Ψ^T x, with x the output of the first call to the inner algorithm
        Vector<> const PsiT_x = DummyAlgorithm::reweightee({}, input.array() + 0.1);
        auto delta = standard_deviation(PsiT_x);
        CHECK(result.weights.array().isApprox(delta / (delta + PsiT_x.array().abs())));
      }
      THEN("The inner algorithm was called twice") {
        CHECK(DummyAlgorithm::called_with_x == 1);
        CHECK(DummyAlgorithm::called_with_warm == 1);
        CHECK(result.inner_loop.x.array().isApprox(input.array() + 0.2));
      }
    }
  }
}
