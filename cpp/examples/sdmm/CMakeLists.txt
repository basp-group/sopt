include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
add_example(euclidian_norm DEPENDS lookup_dependencies LABELS sdmm)
add_example(image NOTEST LIBRARIES sopt wavelets tools_for_tests)
