#include <complex>
#include "catch.hpp"
#include <iostream>
#include <iomanip>

#include "sopt/operators.h"

TEST_CASE("Function Operator", "[ops]") {
  using namespace sopt;

  typedef int SCALAR;
  typedef Eigen::Array<SCALAR, Eigen::Dynamic, 1> t_Vector;

  auto const N = 10;
  auto direct = [](t_Vector &out, t_Vector const&input) { out = input * 2 - 1; };
  auto transpose = [](t_Vector &out, t_Vector const&input) { out = input * 4 - 1; };
  t_Vector const x = t_Vector::Random(2*N) * 5;

  SECTION("Operators") {
    auto op = sopt::make_operator<t_Vector>(direct, transpose);

    CHECK((op * x).matrix() == (2 * x - 1).matrix());
    CHECK((op.transpose() * x).matrix() == (4 * x - 1).matrix());
  }
  SECTION("Proximal operators") {
    auto proximal = [](t_Vector &out, t_Vector const&input) { out = input * 5 - 1; };
    auto op = sopt::make_operator<t_Vector>(direct, transpose, proximal);

    CHECK((op * x).matrix() == (2 * x - 1).matrix());
    CHECK((op.transpose() * x).matrix() == (4 * x - 1).matrix());
    CHECK(op.proximal(x).matrix() == (5 * x - 1).matrix());
  }
}

TEST_CASE("Matrix Operator", "[ops]") {
  using namespace sopt;

  typedef int SCALAR;
  typedef Eigen::Array<SCALAR, Eigen::Dynamic, 1> t_Vector;
  typedef Eigen::Array<SCALAR, Eigen::Dynamic, Eigen::Dynamic> t_Matrix;

  auto const N = 10;
  t_Matrix const L = t_Matrix::Random(N, N);
  t_Vector const x = t_Vector::Random(N) * 5;

  SECTION("Operators") {
    auto op = sopt::make_operator(L.matrix());


    CHECK(op * x.matrix() == L.matrix() * x.matrix());
    CHECK(op.transpose() * x.matrix() == L.transpose().matrix() * x.matrix());
  }
}

TEST_CASE("Array of operators", "[ops]") {
  using namespace sopt;

  typedef int SCALAR;
  typedef Eigen::Matrix<SCALAR, Eigen::Dynamic, 1> t_Vector;
  typedef Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> t_Matrix;

  auto const N = 10;
  t_Vector const x = t_Vector::Random(N) * 5;
  std::vector<t_Matrix> Ls{t_Matrix::Random(N, N), t_Matrix::Random(N, N)};
  std::vector<Operator<t_Vector>> ops;
  for(auto const &matrix: Ls) ops.emplace_back(sopt::make_operator(matrix));

  for(int i(0); i < ops.size(); ++i) {
    CHECK(ops[i] * x == Ls[i] * x);
    CHECK(ops[i].transpose() * x == Ls[i].transpose() * x);
  }
}
