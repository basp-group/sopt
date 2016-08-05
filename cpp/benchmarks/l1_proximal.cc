#include <sstream>
#include <benchmark/benchmark.h>
#include <sopt/l1_proximal.h>
#include <sopt/real_type.h>
#include <sopt/types.h>

template <class TYPE> void function_l1p(benchmark::State &state) {
  typedef typename sopt::real_type<TYPE>::type Real;
  auto const N = state.range_x();
  auto const input = sopt::Vector<TYPE>::Random(N).eval();
  auto const Psi = sopt::Matrix<TYPE>::Random(input.size(), input.size() * 10).eval();
  sopt::Vector<Real> const weights
      = sopt::Vector<TYPE>::Random(Psi.cols()).normalized().array().abs();

  auto const l1 = sopt::proximal::L1<TYPE>()
                      .tolerance(std::pow(10, -state.range_y()))
                      .itermax(100)
                      .fista_mixing(true)
                      .positivity_constraint(true)
                      .nu(1)
                      .Psi(Psi)
                      .weights(weights);

  Real const gamma = 1e-2 / Psi.array().abs().sum();
  auto output = sopt::Vector<TYPE>::Zero(N).eval();
  while(state.KeepRunning())
    l1(output, gamma, input);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(N) * sizeof(TYPE));
}

BENCHMARK_TEMPLATE(function_l1p, sopt::t_complex)->RangePair(1, 256, 4, 12)->UseRealTime();
BENCHMARK_TEMPLATE(function_l1p, sopt::t_real)->RangePair(1, 256, 4, 12)->UseRealTime();

BENCHMARK_MAIN()
