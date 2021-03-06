add_library(tools_for_tests STATIC tiffwrappers.cc tiffwrappers.h)
target_link_libraries(tools_for_tests ${TIFF_LIBRARY})
target_include_directories(tools_for_tests PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  "${PROJECT_BINARY_DIR}/include/"
)
target_include_directories(tools_for_tests SYSTEM PUBLIC ${TIFF_INCLUDE_DIR})
if(SPDLOG_INCLUDE_DIR)
  target_include_directories(tools_for_tests SYSTEM PUBLIC ${SPDLOG_INCLUDE_DIR})
endif()
if(EIGEN3_INCLUDE_DIR)
  target_include_directories(tools_for_tests SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
endif()
add_dependencies(tools_for_tests lookup_dependencies)

# Simple manual tester of read/write tiff
add_executable(copy_tiff copy_tiff.cc)
target_link_libraries(copy_tiff tools_for_tests sopt)

configure_file("directories.in.h" "${PROJECT_BINARY_DIR}/include/tools_for_tests/directories.h")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/outputs")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/outputs/sdmm")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/outputs/sdmm/regressions")
