#include <sopt/maths.h>
#include <sopt/types.h>

int main(int, char const **) {

  // Create a matrix with a single negative real numbers
  typedef sopt::Image<std::complex<int>> t_Matrix;
  t_Matrix input = 2 * t_Matrix::Ones(5, 5) + t_Matrix::Random(5, 5);

  // Apply projection
  t_Matrix posquad = sopt::positive_quadrant(input);
  // imaginary part and negative real part becomes zero
  if((posquad.array().imag() != 0).any())
    throw std::runtime_error("Imaginary part not zero");

  // positive real part unchanged
  posquad.real()(2, 3) = input.real()(2, 3);
  if((posquad.array().real() != input.array().real()).all())
    throw std::runtime_error("Real part was modified");

  return 0;
}
