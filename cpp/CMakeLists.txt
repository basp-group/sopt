# some common compilation flags
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
if(EIGEN3_INCLUDE_DIR)
  include_directories("${EIGEN3_INCLUDE_DIR}")
endif()

# Setup logging
unset(SOPT_DO_LOGGING)
if(logging)
  set(SOPT_DO_LOGGING 1)
endif()
configure_file(sopt/logging.in.h ${EXTERNAL_ROOT}/include/sopt/logging.h)

add_subdirectory(wavelets)
if(tests)
  add_subdirectory(tests)
  add_subdirectory(examples)
endif()
if(benchmarks)
  add_subdirectory(benchmarks)
endif()
