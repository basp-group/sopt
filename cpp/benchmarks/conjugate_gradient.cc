#include <sstream>
#include <benchmark/benchmark.h>
#include "sopt/conjugate_gradient.h"

template<class TYPE> void matrix_cg(benchmark::State &state) {
  auto const N = state.range_x();
  auto const epsilon = std::pow(10, -state.range_y());
  auto const A = sopt::Image<TYPE>::Random(N, N).eval();
  auto const b = sopt::Array<TYPE>::Random(N).eval();

  auto const AhA = A.matrix().transpose().conjugate() * A.matrix();
  auto const Ahb = A.matrix().transpose().conjugate() * b.matrix();
  auto output = sopt::Vector<TYPE>::Zero(N).eval();
  sopt::ConjugateGradient cg(0, epsilon);
  while(state.KeepRunning())
    cg(output, AhA, Ahb);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(N) * sizeof(TYPE));
}

template<class TYPE> void function_cg(benchmark::State &state) {
  auto const N = state.range_x();
  auto const epsilon = std::pow(10, -state.range_y());
  auto const A = sopt::Image<TYPE>::Random(N, N).eval();
  auto const b = sopt::Array<TYPE>::Random(N).eval();

  auto const AhA = A.matrix().transpose().conjugate() * A.matrix();
  auto const Ahb = A.matrix().transpose().conjugate() * b.matrix();
  typedef sopt::RefVector<TYPE> t_RefVector;
  typedef sopt::ConstRefVector<TYPE> t_ConstRefVector;
  auto func = [&AhA](t_RefVector out, t_ConstRefVector const &input) { out = AhA * input; };
  auto output = sopt::Vector<TYPE>::Zero(N).eval();
  sopt::ConjugateGradient cg(0, epsilon);
  while(state.KeepRunning())
    cg(output, func, Ahb);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(N) * sizeof(TYPE));
}

BENCHMARK_TEMPLATE(matrix_cg, sopt::t_complex) ->RangePair(1, 256, 4, 12);
BENCHMARK_TEMPLATE(matrix_cg, sopt::t_real) ->RangePair(1, 256, 4, 12);
BENCHMARK_TEMPLATE(function_cg, sopt::t_complex) ->RangePair(1, 256, 4, 12);
BENCHMARK_TEMPLATE(function_cg, sopt::t_real) ->RangePair(1, 256, 4, 12);

BENCHMARK_MAIN()
