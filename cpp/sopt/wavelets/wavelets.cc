#include "sopt/wavelets/wavelets.h"
#include <exception>

namespace sopt {
namespace wavelets {

Wavelet factory(std::string name, t_uint nlevels) {
  // Dirac not implemented yet
  if(name == "dirac" or name == "Dirac")
    throw std::exception();

  if(name.substr(0, 2) == "DB" or name.substr(0, 2) == "db") {
    std::istringstream sstr(name.substr(2, name.size() - 2));
    t_uint l(0);
    sstr >> l;
    return Wavelet(daubechies_data(l), nlevels);
  }
  // Unknown input wavelet
  throw std::exception();
}
}
}
