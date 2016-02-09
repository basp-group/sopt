#include <exception>
#include <sopt/padmm.h>
#include <sopt/proximal.h>
#include <sopt/relative_variation.h>
#include <sopt/types.h>

// We will minimize ||x - x_0|| + ||x - x_1||, ||.|| the euclidian norm
int main(int, char const **) {
  // Initializes and sets logger (if compiled with logging)
  // See set_level function for levels.
  sopt::logging::initialize();

  // Some typedefs for simplicity
  typedef sopt::t_real t_Scalar;
  typedef sopt::Vector<t_Scalar> t_Vector;
  typedef sopt::Matrix<t_Scalar> t_Matrix;

  // Creates the target vectors
  auto const N = 5;
  t_Vector const target0 = t_Vector::Random(N);
  t_Vector const target1 = t_Vector::Random(N) * 4;

  // Creates the resulting proximal
  // In practice g_0 and g_1 are any functions with the signature
  // void(t_Vector &output, t_Vector::Scalar gamma, t_Vector const &input)
  // They are the proximal of ||x - x_0|| and ||x - x_1||
  auto prox_g0 = sopt::proximal::translate(sopt::proximal::EuclidianNorm(), -target0);
  auto prox_g1 = sopt::proximal::translate(sopt::proximal::EuclidianNorm(), -target1);

  auto padmm = sopt::algorithm::ProximalADMM<t_Scalar>(prox_g0, prox_g1)
                   .itermax(5000)
                   .is_converged(sopt::RelativeVariation<t_Scalar>(1e-12))
                   .gamma(0.01)
                   // Phi == -1, so that we can minimize f(x) + g(x), as per problem definition in
                   // padmm.
                   .Phi(-t_Matrix::Identity(N, N));

  auto const diagnostic = padmm(t_Vector::Zero(N));

  // diagnostic should tell us the function converged
  // it also contains diagnostic.niters - the number of iterations, and cg_diagnostic - the
  // diagnostic from the last call to the conjugate gradient.
  if(not diagnostic.good)
    throw std::runtime_error("Did not converge!");

  // x should be any point on the segment linking x_0 and x_1
  t_Vector const segment = (target1 - target0).normalized();
  t_Scalar const alpha = (diagnostic.x - target0).transpose() * segment;
  if((target1 - target0).transpose() * segment < alpha)
    throw std::runtime_error("Point beyond x_1 plane");
  if(alpha < 0e0)
    throw std::runtime_error("Point before x_0 plane");
  if((diagnostic.x - target0 - alpha * segment).stableNorm() > 1e-8)
    throw std::runtime_error("Point not on (x_0, x_1) line");

  return 0;
}
