#include <sopt/utility.h>
#include <sopt/types.h>

int main(int, char const **) {
  typedef Eigen::Matrix<std::complex<int>, Eigen::Dynamic, Eigen::Dynamic> t_Matrix;
  t_Matrix input(2, 2), weights(2, 2);
  input << 1, -2, 3, -4; weights << 5, 6, 7, 8;

  if(sopt::l1_norm(input, weights) != 1*5 + 2*6 + 3*7 + 4*8)
    throw std::exception();

  return 0;
}
