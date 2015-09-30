add_library(tools_for_tests STATIC tiffwrappers.cc tiffwrappers.h)
target_include_directories(tools_for_tests PUBLIC ${TIFF_INCLUDE_DIR})
target_link_libraries(tools_for_tests ${TIFF_LIBRARY})
add_dependencies(lookup_dependencies tools_for_tests)

# Simple manual tester of read/write tiff
add_executable(copy_tiff copy_tiff.cc)
target_link_libraries(copy_tiff tools_for_tests)

configure_file("directories.in.h" "${EXTERNAL_ROOT}/include/tools_for_tests/directories.h")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/outputs")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/outputs/sdmm")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/outputs/sdmm/regressions")
