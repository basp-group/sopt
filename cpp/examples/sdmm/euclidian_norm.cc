#include <exception>
#include <sopt/sdmm.h>
#include <sopt/types.h>

// We will minimize ||L_0 x - x_0|| + ||L_1 x - x_1||, ||.|| the euclidian norm
int main(int, char const **) {
  // Initializes and sets logger (if compiled with logging)
  // See set_level function for levels.
  sopt::logging::initialize();
  sopt::logging::set_level(SOPT_TEST_DEBUG_LEVEL);

  // Some typedefs for simplicity
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> t_Vector;
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> t_Matrix;

  // Creates the transformation matrices
  auto const N = 10;
  t_Matrix const L0 = t_Matrix::Random(N, N) * 2;
  t_Matrix const L1 = t_Matrix::Random(N, N) * 4;
  // L1_direct and L1_dagger are used to demonstrate that we can define L_i in SDMM both directly
  // as matrices, or as a pair of functions that apply a linear operator and its transpose.
  auto L1_direct = [&L1](t_Vector &out, t_Vector const &input) { out = L1 * input; };
  auto L1_dagger = [&L1](t_Vector &out, t_Vector const &input) {
    out = L1.transpose().conjugate() * input;
  };
  // Creates the target vectors
  t_Vector const target0 = t_Vector::Random(N);
  t_Vector const target1 = t_Vector::Random(N);

  // Creates the resulting proximal
  // In practice g_0 and g_1 are any functions with the signature
  // void(t_Vector &output, t_Vector::Scalar gamma, t_Vector const &input)
  auto prox_g0 = sopt::proximal::translate(sopt::proximal::EuclidianNorm(), -target0);
  auto prox_g1 = sopt::proximal::translate(sopt::proximal::EuclidianNorm(), -target1);

  // This function is called at every iteration. It should return true when convergence is achieved.
  // Otherwise the convex optimizer will trudge on until the requisite number of iterations have
  // been achieved.
  // It takes the convex minimizer and the current candidate output vector as arguments.
  // The example below assumes convergence when the candidate vector does not change anymore.
  typedef sopt::algorithm::SDMM<t_Vector::Scalar> SDMM;
  std::shared_ptr<t_Vector> previous;
  auto relative = [&previous](SDMM const&, t_Vector const &candidate) {
    if(not previous) {
      previous = std::make_shared<t_Vector>(candidate);
      return false;
    }
    auto const norm = (*previous - candidate).stableNorm();
    SOPT_INFO("   - Checking convergence {}", norm);
    auto const result = norm < 1e-8 * candidate.size();
    *previous = candidate;
    return result;
  };

  // Now we can create the sdmm convex minimizer
  // Its parameters are set by calling member functions with appropriate names.
  auto sdmm = SDMM()
    .itermax(500) // maximum number of iterations
    .gamma(1)
    .conjugate_gradient(std::numeric_limits<sopt::t_uint>::max(), 1e-12)
    .is_converged(relative)
    // Any number of (proximal g_i, L_i) pairs can be added
    // L_i can be a matrix
    .append(prox_g0, L0)
    // L_i can be a pair of functions applying a linear transform and its transpose
    .append(prox_g1, L1_direct, L1_dagger);

  t_Vector result;
  t_Vector const input = t_Vector::Random(N);
  auto const diagnostic = sdmm(result, input);

  // diagnostic should tell us the function converged
  // it also contains diagnostic.niters - the number of iterations, and cg_diagnostic - the
  // diagnostic from the last call to the conjugate gradient.
  if(not diagnostic.good)
    throw std::runtime_error("Did not converge!");

  // Lets test we are at a minimum by recreating the objective function
  // and checking that stepping in any direction raises its value
  auto const objective = [&target0, &target1, &L0, &L1](t_Vector const&x) {
    return (L0 * x - target0).stableNorm() + (L1 * x - target1).stableNorm();
  };
  auto const minimum = objective(result);
  for(int i(0); i < N; ++i) {
    t_Vector epsilon = t_Vector::Zero(N);
    epsilon(i) = 1e-4;
    auto const at_x_plus_epsilon = objective(input + epsilon);
    if(minimum >= at_x_plus_epsilon)
      throw std::runtime_error("That's no minimum!");
  }

  return 0;
}
