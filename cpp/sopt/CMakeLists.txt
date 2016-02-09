# list of headers
set(headers
  # sopt headers proper
  conjugate_gradient.h logging.disabled.h proximal.h types.h exception.h logging.enabled.h
  sampling.h utility.h linear_transform.h sdmm.h wrapper.h logging.h
  relative_variation.h padmm.h l1_padmm.h ${EXTERNAL_ROOT}/include/sopt/config.h

  # wavelet headers
  wavelets/direct.h wavelets/innards.impl.h wavelets/wavelet_data.h
  wavelets/indirect.h wavelets/sara.h wavelets/wavelets.h
)

set(sources wavelets/wavelets.cc wavelets/wavelet_data.cc wavelets/sara.cc)

add_library(sopt SHARED ${sources})
add_dependencies(sopt lookup_dependencies)
set_target_properties(sopt PROPERTIES VERSION 2.0.0 SOVERSION 2.0.0)
if(SPDLOG_INCLUDE_DIR)
  set_target_properties(sopt PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${SPDLOG_INCLUDE_DIR})
endif()

install(FILES ${headers} DESTINATION include/sopt)
install(TARGETS sopt
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include/sopt
)
