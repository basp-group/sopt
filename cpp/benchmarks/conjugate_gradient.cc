#include <sstream>
#include <benchmark/benchmark.h>
#include "sopt/conjugate_gradient.h"

template<class TYPE> void matrix_cg(benchmark::State &state) {
  auto const N = state.range_x();
  auto const epsilon = std::pow(10, -state.range_y());
  auto const A = Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic>::Random(N, N).eval();
  auto const b = Eigen::Array<TYPE, Eigen::Dynamic, 1>::Random(N).eval();

  auto const AhA = A.matrix().transpose().conjugate() * A.matrix();
  auto const Ahb = A.matrix().transpose().conjugate() * b.matrix();
  auto output = Eigen::Matrix<TYPE, Eigen::Dynamic, 1>::Zero(N).eval();
  sopt::ConjugateGradient cg(0, epsilon);
  while(state.KeepRunning())
    cg(output, AhA, Ahb);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(N) * sizeof(TYPE));
}

template<class TYPE> void function_cg(benchmark::State &state) {
  auto const N = state.range_x();
  auto const epsilon = std::pow(10, -state.range_y());
  auto const A = Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic>::Random(N, N).eval();
  auto const b = Eigen::Array<TYPE, Eigen::Dynamic, 1>::Random(N).eval();

  auto const AhA = A.matrix().transpose().conjugate() * A.matrix();
  auto const Ahb = A.matrix().transpose().conjugate() * b.matrix();
  typedef Eigen::Matrix<TYPE, Eigen::Dynamic, 1> t_Vector;
  auto func = [&AhA](t_Vector &out, t_Vector const &input) { out = AhA * input; };
  auto output = t_Vector::Zero(N).eval();
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
