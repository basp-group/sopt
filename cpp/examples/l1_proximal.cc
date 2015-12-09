#include <sopt/logging.h>
#include <sopt/l1_proximal.h>
#include <sopt/types.h>

int main(int, char const **) {
  sopt::logging::initialize();
  sopt::logging::set_level(SOPT_TEST_DEBUG_LEVEL);

  typedef sopt::t_complex Scalar;
  typedef sopt::real_type<Scalar>::type Real;
  auto const input = sopt::Vector<Scalar>::Random(10).eval();
  auto const Psi = sopt::Matrix<Scalar>::Random(input.size(), input.size() * 10).eval();
  sopt::Vector<Real> const weights
    = sopt::Vector<Scalar>::Random(Psi.cols()).normalized().array().abs();

  auto const l1 = sopt::proximal::L1<Scalar>()
    .tolerance(1e-12)
    .itermax(100)
    .fista_mixing(true)
    .positivity_constraint(true)
    .nu(1)
    .Psi(Psi)
    .weights(weights);

  // gamma should be sufficiently small. Or is it nu should not be 1?
  // In any case, this seems to work.
  Real const gamma = 1e-2 / Psi.array().abs().sum();
  auto const result = l1(gamma, input);

  if(not result.good)
    SOPT_THROW("Did not converge");

  // Check the proximal is a minimum in any allowed direction (positivity constraint)
  Real const eps = 1e-4;
  for(size_t i(0); i < 10; ++i) {
    sopt::Vector<Scalar> const dir = sopt::Vector<Scalar>::Random(input.size()).normalized() * eps;
    sopt::Vector<Scalar> const position = sopt::positive_quadrant(result.proximal + dir);
    Real const dobj = l1.objective(input, position, gamma);
    // Fuzzy logic
    if(dobj < result.objective - 1e-8)
      SOPT_THROW("This is not the minimum we are looking for: ")
        << dobj << " <~ " << result.objective;
  }


  return 0;
}
