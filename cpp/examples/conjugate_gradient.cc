#include <exception>
#include <sopt/conjugate_gradient.h>
#include <sopt/types.h>

int main(int, char const **) {
  // Conjugate-gradient solves  Ax=b, where A is positive definite.
  // The input to our conjugate gradient can be a matrix or a function
  // Lets try both approaches.

  // Creates the input.
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> t_cVector;
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> t_cMatrix;
  t_cVector const b = t_cVector::Random(8);
  t_cMatrix const A = t_cMatrix::Random(b.size(), b.size());

  // Transform to solvable problem A^hA x = A^hb, where A^h is the conjugate transpose
  t_cMatrix const AhA = A.conjugate().transpose() * A;
  t_cVector const Ahb = A.conjugate().transpose() * b;
  // The same transform can be realised using a function, where out = A^h * A * input.
  // This will recompute AhA every time the function is applied by the conjugate gradient. It is not
  // optmial for this case. But the function interface means A could be an FFT.
  auto aha_function = [&A](t_cVector &out, t_cVector const &input) {
    out = A.conjugate().transpose() * A * input;
  };

  // Conjugate gradient with unlimited iterations and a convergence criteria of 1e-12
  sopt::ConjugateGradient cg(0, 1e-12);

  // Call conjugate gradient using both approaches
  auto as_matrix = cg(AhA, Ahb);
  auto as_function = cg(aha_function, Ahb);

  // Check result
  if(not (as_matrix.good and as_function.good))
    throw std::runtime_error("Expected convergence");
  if(as_matrix.niters != as_function.niters)
    throw std::runtime_error("Expected same number of iterations");
  if(as_matrix.residual > cg.tolerance() or as_function.residual > cg.tolerance())
    throw std::runtime_error("Expected better convergence");
  if(not as_matrix.result.isApprox(as_function.result, 1e-6))
    throw std::runtime_error("Expected same result");
  if(not (A * as_matrix.result).isApprox(b, 1e-6))
    throw std::runtime_error("Expected solution to Ax=b");

  return 0;
}
