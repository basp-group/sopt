set(DATA_DIRECTORY "${PROJECT_SOURCE_DIR}/images")
configure_file("datadir.in.h" "${EXTERNAL_ROOT}/include/sopt/datadir.h")

include_directories(${TIFF_INCLUDE_DIR})
add_library(tiffutils STATIC tiffwrappers.cc tiffwrappers.h)
target_link_libraries(tiffutils ${TIFF_LIBRARY})

add_example(wavelets LIBRARIES wavelets LABELS wavelets)
add_example(sara LIBRARIES wavelets LABELS wavelets)
add_example(positive_quadrant_projection DEPENDS lookup_dependencies LABELS proximals)
add_example(l1_norm DEPENDS lookup_dependencies LABELS proximals)
add_example(soft_threshhold DEPENDS lookup_dependencies LABELS proximals)
add_example(conjugate_gradient DEPENDS lookup_dependencies)

# Simple manual tester of read/write tiff
add_executable(copy_tiff copy_tiff.cc)
target_link_libraries(copy_tiff sopt wavelets tiffutils)

add_subdirectory(sdmm)
