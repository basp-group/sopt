include(PackageLookup)  # check for existence, or install external projects

lookup_package(Eigen3 REQUIRED)
if(logging)
  lookup_package(spdlog REQUIRED)
endif()
