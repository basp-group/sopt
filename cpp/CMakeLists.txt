# some common compilation flags
include_directories("${CMAKE_CURRENT_SOURCE_DIR}" "${EIGEN3_INCLUDE_DIR}")

add_subdirectory(wavelets)
if(tests)
  add_subdirectory(tests)
endif()
