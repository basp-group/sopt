#include <sopt/wavelets/sara.h>

int main(int, char const **) {

  // Creates SARA with two wavelets
  typedef std::tuple<std::string, sopt::t_uint> t_i;
  sopt::wavelets::SARA sara{t_i{"DB4", 5}, t_i{"DB8", 2}};

  // Then another one for good measure
  sara.emplace_back("DB3", 7);

  // Creates a random signal
  sopt::Image<sopt::t_complex> input = sopt::Image<sopt::t_complex>::Random(128, 128);
  // Now gets its coefficients
  auto coefficients = sara.direct(input);
  // And transform back. We pass a pre-defined matrix explicitly to illustrate that API.
  // But we could just store the return value as above.
  sopt::Image<sopt::t_complex> recover; // This matrix will be resized if necessary
  sara.indirect(coefficients, recover);

  // Check the reconstruction is corrrect
  if(not input.isApprox(recover))
    throw std::exception();

  // The coefficient for each wavelet basis is stored alongs columns:
  sopt::Image<sopt::t_complex> const DB3_coeffs = sara[2].direct(input) / std::sqrt(sara.size());
  if(not coefficients.rightCols(input.cols()).isApprox(DB3_coeffs))
    throw std::exception();

  return 0;
}
