# list of headers
set(headers
  conjugate_gradient.h logging.disabled.h proximal.h types.h exception.h logging.enabled.h
  sampling.h utility.h linear_transform.h sdmm.h wrapper.h ${EXTERNAL_ROOT}/include/sopt/logging.h
  relative_variation.h wavelets.h)

add_library(sopt sampling.cc ${headers})
add_dependencies(sopt lookup_dependencies)

install(FILES ${headers} DESTINATION include/sopt)
install(TARGETS sopt
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include/sopt
)
