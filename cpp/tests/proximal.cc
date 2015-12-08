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
    typedef t_complex comp;
    Vector<t_complex> const p = l1(gamma, x);
    auto const mini = l1.objective(x, p, gamma);
    auto const eps = 1e-4;
    for(Vector<t_complex>::Index i(0); i < p.size(); ++i) {
      for(auto const dir: {comp(eps, 0), comp(0, eps), comp(-eps, 0), comp(0, -eps)}) {
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
    l1.weights(Vector<t_real>::Random(input.size()));
    check_is_minimum(input, 0.235);
  }

  SECTION("Psi is a concatenation of permutations") {
    auto const psi = concatenated_permutations<t_complex>(input.size(), input.size() * 10);
    l1.Psi(psi).weights(1e0);
    check_is_minimum(input, 0.235);
  }
}
