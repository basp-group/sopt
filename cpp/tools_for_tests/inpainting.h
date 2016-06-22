#ifndef SOPT_TOOLS_FOR_TESTS_INPAINTING_H
#define SOPT_TOOLS_FOR_TESTS_INPAINTING_H

#include "sopt/config.h"
#include <random>
#include <sopt/linear_transform.h>
#include <sopt/types.h>
#include "sopt_l1.h"

namespace sopt {

template <class T>
Vector<T> target(sopt::LinearTransform<Vector<T>> const &sampling, sopt::Image<T> const &image) {
  return sampling * Vector<T>::Map(image.data(), image.size());
}

template <class T>
typename real_type<T>::type
sigma(sopt::LinearTransform<Vector<T>> const &sampling, sopt::Image<T> const &image) {
  auto const snr = 30.0;
  auto const y0 = target(sampling, image);
  return y0.stableNorm() / std::sqrt(y0.size()) * std::pow(10.0, -(snr / 20.0));
}

template <class T, class RANDOM>
Vector<T> dirty(sopt::LinearTransform<Vector<T>> const &sampling, sopt::Image<T> const &image,
                RANDOM &mersenne) {

  // values near the mean are the most likely
  // standard deviation affects the dispersion of generated values from the mean
  auto const y0 = target(sampling, image);
  std::normal_distribution<> gaussian_dist(0, sigma(sampling, image));
  Vector<T> y(y0.size());
  for(t_int i = 0; i < y0.size(); i++)
    y(i) = y0(i) + gaussian_dist(mersenne);

  return y;
}

template <class T>
typename real_type<T>::type
epsilon(sopt::LinearTransform<Vector<T>> const &sampling, sopt::Image<T> const &image) {
  auto const y0 = target(sampling, image);
  auto const nmeasure = y0.size();
  return std::sqrt(nmeasure + 2 * std::sqrt(nmeasure)) * sigma(sampling, image);
}
} /* sopt  */
#endif
