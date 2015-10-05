# some common compilation flags
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
if(EIGEN3_INCLUDE_DIR)
  include_directories("${EIGEN3_INCLUDE_DIR}")
endif()

# Setup logging
unset(SOPT_DO_LOGGING)
if(logging)
  include_directories(${SPDLOG_INCLUDE_DIR})
  set(SOPT_DO_LOGGING 1)
  set(SOPT_TEST_LOG_LEVEL off CACHE STRING "Level when logging tests")
  set_property(CACHE SOPT_TEST_LOG_LEVEL PROPERTY STRINGS
    off trace debug info notice warn err critical alert emerg)
  add_definitions(-DSOPT_TEST_DEBUG_LEVEL="${SOPT_TEST_LOG_LEVEL}")
endif()
configure_file(sopt/logging.in.h ${EXTERNAL_ROOT}/include/sopt/logging.h)
include_directories("${EXTERNAL_ROOT}/include/")

add_subdirectory(wavelets)
if(tests)
  add_subdirectory(tests)
  add_subdirectory(examples)
endif()
if(benchmarks)
  add_subdirectory(benchmarks)
endif()
