#include <initializer_list>
#include "wavelets.h"
#include "traits.h"

// helper convolution functions
// kept in anonymous namespace so as to not pollute the rest of the library
// this file should not be included explcitly except here on in tests.
#include "convolve.impl.cc"

namespace sopt { namespace wavelets {
  // Function inside anonymouns namespace won't appear in library
  namespace {
    //! Vector setup from initializer list, because easier
    WaveletData::t_vector init_db(std::initializer_list<t_real> const &input) {
      Eigen::Matrix<t_real, Eigen::Dynamic, 1> result(input.size());
      std::copy(input.begin(), input.end(), result.data());
      return result;
    }
    //! Every other element is negative
    WaveletData::t_vector negate_odd(WaveletData::t_vector const &coeffs) {
      WaveletData::t_vector result(coeffs);
      for(t_int i(0); i < static_cast<t_int>(coeffs.size()); ++i)
        result(i) -= result(i);
      return result;
    }
    //! Odd elements only
    WaveletData::t_vector odd(WaveletData::t_vector const &coeffs) {
      WaveletData::t_vector result(coeffs.size() >> 1);
      for(t_int i(1); i < static_cast<t_int>(result.size()); i+=2)
        result(i >> 1) = coeffs(i);
      return result;
    }
    //! Even elements only
    WaveletData::t_vector even(WaveletData::t_vector const &coeffs) {
      WaveletData::t_vector result((coeffs.size() + 1) >> 1);
      for(t_int i(0); i < static_cast<t_int>(result.size()); i += 2)
        result(i >> 1) = coeffs(i);
      return result;
    }
  }

  WaveletData::WaveletData(std::initializer_list<t_scalar> const &coeffs)
    : WaveletData(init_db(coeffs)) {}

  WaveletData::WaveletData(t_vector const &coeffs)
    : coefficients(coeffs), direct_filter({coeffs, negate_odd(coeffs).reverse()}),
    indirect_filter({even(coeffs.reverse()), odd(coeffs.reverse()), even(coeffs), -odd(coeffs)}) {}

  const WaveletData Daubechies1({0.707106781186548, 0.707106781186548e0});
  const WaveletData Daubechies2({0.482962913144690, 0.836516303737469, 0.224143868041857,
      -0.129409522550921});
  const WaveletData Daubechies3({0.332670552950957, 0.806891509313339, 0.459877502119331,
      -0.135011020010391, -0.085441273882241, 0.035226291882101});
  const WaveletData Daubechies4({0.230377813308855, 0.714846570552542, 0.630880767929590,
      -0.027983769416984, -0.187034811718881, 0.030841381835987, 0.032883011666983,
      -0.010597401784997});
  const WaveletData Daubechies5({0.160102397974125, 0.603829269797473, 0.724308528438574,
      0.138428145901103, -0.242294887066190, -0.032244869585030, 0.077571493840065,
      -0.006241490213012, -0.012580751999016, 0.003335725285002});
  const WaveletData Daubechies6({0.111540743350080, 0.494623890398385, 0.751133908021578,
      0.315250351709243, -0.226264693965169, -0.129766867567096, 0.097501605587079,
      0.027522865530016, -0.031582039318031, 0.000553842200994, 0.004777257511011,
      -0.001077301084996});
  const WaveletData Daubechies7({0.077852054085062, 0.396539319482306, 0.729132090846555,
      0.469782287405359, -0.143906003929106, -0.224036184994166, 0.071309219267050,
      0.080612609151066, -0.038029936935035, -0.016574541631016, 0.012550998556014,
      0.000429577973005, -0.001801640704000, 0.000353713800001});
  const WaveletData Daubechies8({0.054415842243082, 0.312871590914466, 0.675630736298013,
      0.585354683654869, -0.015829105256024, -0.284015542962428, 0.000472484573998,
      0.128747426620186, -0.017369301002022, -0.044088253931065, 0.013981027917016,
      0.008746094047016, -0.004870352993011, -0.000391740372996, 0.000675449405999,
      -0.000117476784002});
  const WaveletData Daubechies9({0.038077947363167, 0.243834674637667, 0.604823123676779,
      0.657288078036639, 0.133197385822089, -0.293273783272587, -0.096840783220879,
      0.148540749334760, 0.030725681478323, -0.067632829059524, 0.000250947114992,
      0.022361662123515, -0.004723204757895, -0.004281503681905, 0.001847646882961,
      0.000230385763995, -0.000251963188998, 0.000039347319995});
  const WaveletData Daubechies10({0.026670057900951, 0.188176800077621, 0.527201188930920,
      0.688459039452592, 0.281172343660426, -0.249846424326489, -0.195946274376597,
      0.127369340335743, 0.093057364603807, -0.071394147165861, -0.029457536821946,
      0.033212674058933, 0.003606553566988, -0.010733175482980, 0.001395351746994,
      0.001992405294991, -0.000685856695005, -0.000116466854994, 0.000093588670001,
      -0.000013264203002});
}}
