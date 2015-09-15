# some common compilation flags
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
if(EIGEN3_INCLUDE_DIR)
  include_directories("${EIGEN3_INCLUDE_DIR}")
endif()


add_subdirectory(wavelets)
if(tests)
  add_subdirectory(tests)
  add_subdirectory(examples)
endif()
if(benchmarks)
  add_subdirectory(benchmarks)
endif()
