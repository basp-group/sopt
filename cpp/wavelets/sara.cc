#include <exception>
#include "sara.h"

namespace sopt { namespace wavelets {

SARA::SARA(std::initializer_list<std::tuple<std::string, t_uint>> const& init)
  : std::vector<Wavelet>() {
  reserve(init.size());
  for(auto const inputs: init)
    emplace_back(std::get<0>(inputs), std::get<1>(inputs));
}

void SARA::emplace_back(std::string const &name, t_uint nlevels) {
  std::vector<Wavelet>::emplace_back(std::move(factory(name, nlevels)));
}

}}
