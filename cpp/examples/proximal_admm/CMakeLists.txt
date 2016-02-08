include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
add_example(padmm_euclidian_norm euclidian_norm.cc DEPENDS lookup_dependencies LABELS padmm)
# add_example(inpainting LIBRARIES sopt wavelets tools_for_tests)
