#include <wavelets/wavelets.h>

int main(int, char const **) {

  // Creates Daubechies 4 wavelet, with 5 levels
  auto const wavelets = sopt::wavelets::factory("DB4", 5);

  // Creates a random signal
  sopt::Image<sopt::t_complex> input = sopt::Image<sopt::t_complex>::Random(128, 128);
  // Now gets its coefficients
  auto coefficients = wavelets.direct(input);
  // And transform back
  auto recover = wavelets.indirect(coefficients);
  // Check the reconstruction is corrrect
  if(not input.isApprox(recover))
    throw std::exception();

  // To save on memory allocation we could also use a pre-allocated matrix;
  coefficients.fill(0);
  recover.fill(0);
  wavelets.direct(coefficients, input);
  wavelets.indirect(coefficients, recover);
  if(not input.isApprox(recover))
    throw std::exception();

  // Finally, it is possible to use expressions
  // For instance, we can do a 1d transform on a single row
  wavelets.direct(coefficients.row(2).transpose(), input.row(2).transpose() * 2);
  wavelets.indirect(coefficients.row(2).transpose(), recover.row(2).transpose());
  if(not input.row(2).isApprox(recover.row(2) * 0.5))
    throw std::exception();

  return 0;
}
