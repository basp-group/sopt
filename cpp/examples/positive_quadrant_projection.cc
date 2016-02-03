#include <sopt/types.h>
#include <sopt/utility.h>

int main(int, char const **) {

  // Create a matrix with a single negative real numbers
  typedef sopt::Image<std::complex<int>> t_Matrix;
  t_Matrix input = t_Matrix::Ones(5, 5) + t_Matrix::Random(5, 5);
  input.real()(2, 3) *= -1;

  // Apply projection
  t_Matrix posquad = sopt::positive_quadrant(input);
  // imaginary part and negative real part becomes zero
  if((posquad.array().imag() != 0).any() or posquad(2, 3) != 0)
    throw std::exception();

  // positive real part unchanged
  posquad.real()(2, 3) = input.real()(2, 3);
  if((posquad.array().real() != input.array().real()).all())
    throw std::exception();

  return 0;
}
