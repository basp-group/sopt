#include <sstream>
#include <benchmark/benchmark.h>
#include "wavelets/wavelets.h"

unsigned get_size(unsigned requested, unsigned levels) {
  auto const N = (1u << levels);
  return requested % N == 0? requested: requested + N - requested % N;
}
std::string get_name(unsigned db) {
  std::ostringstream sstr; sstr << "DB" << db;
  return sstr.str();
}

template<class TYPE, unsigned DB=1, unsigned LEVEL=1>
void direct_matrix(benchmark::State &state) {
  auto const Nx = get_size(state.range_x(), LEVEL);
  auto const Ny = get_size(state.range_y(), LEVEL);
  auto const input = Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic>::Random(Nx, Ny).eval();
  auto output = Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic>::Zero(Nx, Ny).eval();
  auto const wavelet = sopt::wavelets::factory(get_name(DB), LEVEL);
  while(state.KeepRunning())
    wavelet.direct(output, input);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(Nx) * int64_t(Ny) * sizeof(TYPE));
}

template<class TYPE, unsigned DB=1, unsigned LEVEL=1>
void indirect_matrix(benchmark::State &state) {
  auto const Nx = get_size(state.range_x(), LEVEL);
  auto const Ny = get_size(state.range_y(), LEVEL);
  auto const input = Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic>::Random(Nx, Ny).eval();
  auto output = Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic>::Zero(Nx, Ny).eval();
  auto const wavelet = sopt::wavelets::factory(get_name(DB), LEVEL);
  while(state.KeepRunning())
    wavelet.indirect(input, output);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(Nx) * int64_t(Ny) * sizeof(TYPE));
}

template<class TYPE, unsigned DB=1, unsigned LEVEL=1>
void direct_vector(benchmark::State &state) {
  auto const Nx = get_size(state.range_x(), LEVEL);
  auto const input = Eigen::Array<TYPE, Eigen::Dynamic, 1>::Random(Nx).eval();
  auto output = Eigen::Array<TYPE, Eigen::Dynamic, 1>::Zero(Nx).eval();
  auto const wavelet = sopt::wavelets::factory(get_name(DB), LEVEL);
  while(state.KeepRunning())
    wavelet.direct(output, input);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(Nx) * sizeof(TYPE));
}
template<class TYPE, unsigned DB=1, unsigned LEVEL=1>
void indirect_vector(benchmark::State &state) {
  auto const Nx = get_size(state.range_x(), LEVEL);
  auto const input = Eigen::Array<TYPE, Eigen::Dynamic, 1>::Random(Nx).eval();
  auto output = Eigen::Array<TYPE, Eigen::Dynamic, 1>::Zero(Nx).eval();
  auto const wavelet = sopt::wavelets::factory(get_name(DB), LEVEL);
  while(state.KeepRunning())
    wavelet.indirect(input, output);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(Nx) * sizeof(TYPE));
}

auto const n = 64;
auto const N = 256*3;

BENCHMARK_TEMPLATE(direct_matrix, sopt::t_complex,  1, 1) ->RangePair(n, N, n, N);
BENCHMARK_TEMPLATE(direct_matrix, sopt::t_real,     1, 1) ->RangePair(n, N, n, N);
BENCHMARK_TEMPLATE(direct_matrix, sopt::t_complex, 10, 1) ->RangePair(n, N, n, N);

BENCHMARK_TEMPLATE(direct_vector, sopt::t_complex,  1, 1) ->Range(n, N);
BENCHMARK_TEMPLATE(direct_vector, sopt::t_complex, 10, 1) ->Range(n, N);
BENCHMARK_TEMPLATE(direct_vector, sopt::t_complex,  1, 2) ->Range(n, N);
BENCHMARK_TEMPLATE(direct_vector, sopt::t_real,     1, 1) ->Range(n, N);

BENCHMARK_TEMPLATE(indirect_matrix, sopt::t_complex,  1, 1) ->RangePair(n, N, n, N);
BENCHMARK_TEMPLATE(indirect_matrix, sopt::t_real,     1, 1) ->RangePair(n, N, n, N);
BENCHMARK_TEMPLATE(indirect_matrix, sopt::t_complex, 10, 1) ->RangePair(n, N, n, N);

BENCHMARK_TEMPLATE(indirect_vector, sopt::t_complex,  1, 1) ->Range(n, N);
BENCHMARK_TEMPLATE(indirect_vector, sopt::t_complex, 10, 1) ->Range(n, N);
BENCHMARK_TEMPLATE(indirect_vector, sopt::t_complex,  1, 2) ->Range(n, N);
BENCHMARK_TEMPLATE(indirect_vector, sopt::t_real,     1, 1) ->Range(n, N);

BENCHMARK_MAIN()
