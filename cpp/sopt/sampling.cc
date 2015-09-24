#include "sopt/sampling.h"

namespace sopt {
  Sampling::Sampling(t_uint size, t_uint samples) : indices(size) {
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
    indices.resize(samples);
  }
} /* sopt  */ 
