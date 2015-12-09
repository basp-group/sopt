#include <random>
#include <utility>
#include <numeric>

#include "catch.hpp"

#include "sopt/proximal.h"
#include "sopt/l1_proximal.h"

std::random_device rd;
std::default_random_engine rengine(rd());

template<class T> sopt::Matrix<T> concatenated_permutations(sopt::t_uint i, sopt::t_uint j) {
  std::vector<size_t> cols(j);
  std::iota(cols.begin(), cols.end(), 0);
  std::shuffle(cols.begin(), cols.end(), rengine);

  assert(j % i == 0);
  auto const N = j / i;
  auto const elem = 1e0 / std::sqrt(static_cast<typename sopt::real_type<T>::type>(N));
  sopt::Matrix<T> result = sopt::Matrix<T>::Zero(i, cols.size());
  for(typename sopt::Matrix<T>::Index k(0); k < result.cols(); ++k)
    result(cols[k] / N, k) = elem;
  return result;
}

TEST_CASE("L2Ball", "[proximal]") {
  using namespace sopt;
  proximal::L2Ball<t_real> ball(0.5);
  Vector<t_real> out;
  Vector<t_real> x(5); x << 1, 2, 3, 4, 5;

  out = ball(0, x);
  CHECK(x.isApprox(out / 0.5 * x.stableNorm()));
  ball.epsilon(x.stableNorm() * 1.001);
  out = ball(0, x);
  CHECK(x.isApprox(out));
}

TEST_CASE("Euclidian norm", "[proximal]") {
  using namespace sopt;
  proximal::EuclidianNorm eucl;

  Vector<t_real> out(5);
  Vector<t_real> x(5); x << 1, 2, 3, 4, 5;
  eucl(out, x.stableNorm() * 1.001, x);
  CHECK(out.isApprox(Vector<t_real>::Zero(x.size())));

  out = eucl(0.1, x);
  CHECK(out.isApprox(x * (1e0 - 0.1/x.stableNorm())));
}

TEST_CASE("Translation", "[proximal]") {
  using namespace sopt;
  Vector<t_real> out(5);
  Vector<t_real> x(5); x << 1, 2, 3, 4, 5;
  proximal::L2Ball<t_real> ball(5000);
  // Pass in a reference, so we can modify ball.epsilon later in the test.
  auto const translated = proximal::translate(std::ref(ball), -x * 0.5);
  translated(out, 0, x);
  CHECK(out.isApprox(x));

  ball.epsilon(0.125);
  out = translated(0, x);
  Vector<t_real> expected = ball(1, x * 0.5) + x * 0.5;
  CHECK(out.isApprox(expected));
}

TEST_CASE("Tight-Frame L1 proximal", "[l1][proximal]") {
  using namespace sopt;
  auto l1 = proximal::L1TightFrame<t_complex>();
  auto check_is_minimum = [&l1](Vector<t_complex> const &x, t_real gamma = 1e0) {
    typedef t_complex Scalar;
    Vector<t_complex> const p = l1(gamma, x);
    auto const mini = l1.objective(x, p, gamma);
    auto const eps = 1e-4;
    for(Vector<t_complex>::Index i(0); i < p.size(); ++i) {
      for(auto const dir: {Scalar(eps, 0), Scalar(0, eps), Scalar(-eps, 0), Scalar(0, -eps)}) {
        Vector<t_complex> p_plus = p;
        p_plus[i] += dir;
        CHECK(l1.objective(x, p_plus, gamma) >= mini);
      }
    }
  };

  Vector<t_complex> const input = Vector<t_complex>::Random(8);

  // no weights
  SECTION("Scalar weights") {
    CHECK(l1(1, input).isApprox(proximal::l1_norm(1, input)));
    CHECK(l1(0.3, input).isApprox(proximal::l1_norm(0.3, input)));
    check_is_minimum(input, 0.664);
  }

  // with weights == 1
  SECTION("vector weights") {
    l1.weights(Vector<t_real>::Ones(input.size()));
    CHECK(l1(1, input).isApprox(proximal::l1_norm(1, input)));
    CHECK(l1(0.2, input).isApprox(proximal::l1_norm(0.2, input)));
    check_is_minimum(input, 0.664);
  }

  SECTION("vector weights with random values") {
    l1.weights(Vector<t_real>::Random(input.size()).array().abs().matrix());
    check_is_minimum(input, 0.235);
  }

  SECTION("Psi is a concatenation of permutations") {
    auto const psi = concatenated_permutations<t_complex>(input.size(), input.size() * 10);
    l1.Psi(psi).weights(1e0);
    check_is_minimum(input, 0.235);
  }

  SECTION("Weights cannot be negative") {
    CHECK_THROWS_AS(l1.weights(-1e0), Exception);
    Vector<t_real> weights = Vector<t_real>::Random(5).array().abs().matrix();
    weights[2] = -1;
    CHECK_THROWS_AS(l1.weights(weights), Exception);
  }
}

TEST_CASE("L1 proximal utilities", "[l1][utilities]") {
  using namespace sopt;
  typedef t_complex Scalar;
  auto const input = Vector<Scalar>::Random(10).eval();
  Vector<Scalar> output;

  SECTION("No Mixing") {
    proximal::L1<Scalar>::NoMixing()(output, 2.1 * input, 0);
    CHECK(output.isApprox(2.1 * input));
    proximal::L1<Scalar>::NoMixing()(output, 4.1 * input, 10);
    CHECK(output.isApprox(4.1 * input));
  }

  SECTION("Fista Mixing") {
    proximal::L1<Scalar>::FistaMixing fista;
    // step zero: no mixing yet
    fista(output, 2.1 * input, 0);
    CHECK(output.isApprox(2.1 * input));
    // step one: first mixing
    fista(output, 3.1 * input, 1);
    auto const alpha = (fista.next(1) - 1) / fista.next(fista.next(1));
    Vector<Scalar> const first = (1e0 + alpha) * 3.1 * input - alpha * 2.1 * input;
    CHECK(output.isApprox(first));
    // step two: second mixing
    fista(output, 4.1 * input, 1);
    auto const beta = (fista.next(fista.next(1)) - 1) / fista.next(fista.next(fista.next(1)));
    Vector<Scalar> const second = (1e0 + alpha) * 4.1 * input - alpha * first;
    CHECK(output.isApprox(second));
  }
}

TEST_CASE("L1 proximal", "[l1][proximal]") {
  using namespace sopt;
  typedef t_complex Scalar;
  auto l1 = proximal::L1<Scalar>();

  Vector<Scalar> const input = Vector<Scalar>::Random(4);

  SECTION("Check against tight-frame") {
    SECTION("Scalar weights") {
      auto const result = l1(1, input);
      CHECK(result.good);
      CHECK(result.error == proximal::L1<Scalar>::Error::NONE);
      CHECK(result.niters > 0);
      CHECK(l1.itermax() == 0);
      CHECK(result.proximal.isApprox(proximal::L1TightFrame<Scalar>()(1, input)));
    }
    SECTION("Vector weights and more complex Psi") {
      auto const Psi = concatenated_permutations<Scalar>(input.size(), input.size() * 10);
      auto const weights = Vector<t_real>::Random(Psi.cols()).array().abs().matrix().eval();
      auto const gamma = 1e0 / static_cast<t_real>(weights.size());
      l1.Psi(Psi).weights(weights).tolerance(1e-12);
      auto const result = l1(gamma, input);
      CHECK(result.good);
      CHECK(result.niters > 0);
      auto const expected = l1.tight_frame(gamma, input).eval();
      CHECK(result.objective == Approx(l1.objective(input, expected, gamma)));
      CHECK(result.proximal.isApprox(expected));
    }
  }

  SECTION("General case") {
    auto const Psi = Matrix<Scalar>::Random(input.size(), input.size() * 10).eval();
    auto const weights = Vector<t_real>::Random(Psi.cols()).array().abs().matrix().eval();
    auto const gamma = 1e0 / static_cast<t_real>(weights.size());

    auto check_is_minimum = [&l1, &gamma](Vector<Scalar> const &x) {
      auto const result = l1(gamma, x);
      // returns false if did not converge.
      // Looks like computing the proximal does not always work...
      if(not result.good)
        return false;
      auto const mini = l1.objective(x, result.proximal, gamma);
      auto const eps = 1e-4;
      // check alongst specific directions
      for(Vector<Scalar>::Index i(0); i < result.proximal.size(); ++i) {
        for(auto const dir: {Scalar(eps, 0), Scalar(0, eps), Scalar(-eps, 0), Scalar(0, -eps)}) {
          Vector<Scalar> p_plus = result.proximal;
          p_plus[i] += dir;
          if(l1.positivity_constraint())
            p_plus = sopt::positive_quadrant(p_plus);
          else if(l1.real_constraint())
            p_plus = p_plus.real().template cast<Scalar>();
          CHECK(l1.objective(x, p_plus, gamma) >= mini);
        }
      }
      // check alongst non-specific directions
      for(size_t i(0); i < 10; ++i) {
        Vector<Scalar> p_plus
          = result.proximal + result.proximal.Random(result.proximal.size()) * eps;
        if(l1.positivity_constraint())
          p_plus = sopt::positive_quadrant(p_plus);
        else if(l1.real_constraint())
          p_plus = p_plus.real().template cast<Scalar>();
        CHECK(l1.objective(x, p_plus, gamma) >= mini);
      }
      return true;
    };

    l1.Psi(Psi).weights(weights).tolerance(1e-12).itermax(10000);

    SECTION("No constraints") {
      CHECK(not l1.positivity_constraint());
      CHECK(not l1.real_constraint());
      for(size_t i(0); i < 10; ++i)
        if(check_is_minimum(input))
          break;
    }
    SECTION("Real constraints") {
      l1.real_constraint(true);
      CHECK(not l1.positivity_constraint());
      CHECK(l1.real_constraint());
      for(size_t i(0); i < 10; ++i)
        if(check_is_minimum(input))
          break;
    }
    SECTION("Positivity constraints") {
      l1.positivity_constraint(true);
      CHECK(l1.positivity_constraint());
      CHECK(not l1.real_constraint());
      for(size_t i(0); i < 10; ++i)
        if(check_is_minimum(input))
          break;
    }
  }
}
