#include <catch.hpp>
#include <complex>
#include <iomanip>

#include "sopt/linear_transform.h"

TEST_CASE("Linear Transforms", "[ops]") {
  using namespace sopt;

  typedef int SCALAR;
  typedef Array<SCALAR> t_Vector;
  typedef Image<SCALAR> t_Matrix;
  auto const N = 10;

  SECTION("Functions") {

    auto direct = [](t_Vector &out, t_Vector const &input) { out = input * 2 - 1; };
    auto indirect = [](t_Vector &out, t_Vector const &input) { out = input * 4 - 1; };
    t_Vector const x = t_Vector::Random(2 * N) * 5;

    auto op = sopt::linear_transform<t_Vector>(direct, indirect);

    CHECK((op * x).eval().cols() == x.cols());
    CHECK((op * x).eval().rows() == x.rows());
    CHECK((op * x).cols() == x.cols());
    CHECK((op * x).rows() == x.rows());
    CHECK((op * x).matrix() == (2 * x - 1).matrix());
    CHECK((op.adjoint() * x).matrix() == (4 * x - 1).matrix());
  }

  SECTION("Matrix") {
    t_Matrix const L = t_Matrix::Random(N, N);
    t_Vector const x = t_Vector::Random(N) * 5;

    auto op = sopt::linear_transform(L.matrix());

    CHECK((op * x.matrix()).eval().cols() == x.cols());
    CHECK((op * x.matrix()).eval().rows() == x.rows());
    CHECK((op * x.matrix()).cols() == x.cols());
    CHECK((op * x.matrix()).rows() == x.rows());
    CHECK(op * x.matrix() == L.matrix() * x.matrix());
    CHECK(op.adjoint() * x.matrix() == L.conjugate().transpose().matrix() * x.matrix());
  }

  SECTION("Rectangular matrix") {
    t_Matrix const L = t_Matrix::Random(N, 2 * N);
    t_Vector const x = t_Vector::Random(2 * N) * 5;

    auto op = sopt::linear_transform(L.matrix());

    CHECK((op * x.matrix()).eval().cols() == 1);
    CHECK((op * x.matrix()).eval().rows() == N);
    CHECK((op * x.matrix()).cols() == 1);
    CHECK((op * x.matrix()).rows() == N);
    CHECK(op * x.matrix() == L.matrix() * x.matrix());
    CHECK(op.adjoint() * x.head(N).matrix()
          == L.conjugate().transpose().matrix() * x.head(N).matrix());
  }
}

TEST_CASE("Array of Linear transforms", "[ops]") {
  using namespace sopt;

  typedef int SCALAR;
  typedef Vector<SCALAR> t_Vector;
  typedef Matrix<SCALAR> t_Matrix;

  auto const N = 10;
  t_Vector const x = t_Vector::Random(N) * 5;
  std::vector<t_Matrix> Ls{t_Matrix::Random(N, N), t_Matrix::Random(N, N)};
  std::vector<LinearTransform<t_Vector>> ops;
  for(auto const &matrix : Ls)
    ops.emplace_back(sopt::linear_transform(matrix));

  for(decltype(Ls)::size_type i(0); i < ops.size(); ++i) {
    CHECK(ops[i] * x == Ls[i] * x);
    CHECK(ops[i].adjoint() * x == Ls[i].conjugate().transpose() * x);
  }
}
