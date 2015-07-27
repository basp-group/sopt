add_library(wavelets SHARED wavelets.cc wavelet_data.cc sara.cc)
# if Eigen3 automatically downloaded, then need to add it as dependency
if(TARGET Eigen3)
  add_dependencies(wavelets Eigen3)
endif()
