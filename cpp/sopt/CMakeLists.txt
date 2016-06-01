# list of headers
set(headers
  conjugate_gradient.h logging.disabled.h proximal.h types.h exception.h logging.enabled.h
  sampling.h utility.h linear_transform.h sdmm.h wrapper.h logging.h
  relative_variation.h padmm.h l1_padmm.h ${EXTERNAL_ROOT}/include/sopt/config.h
  real_type.h l1_proximal.h proximal_expression.h wavelets.h
)
set(wavelet_headers
  wavelets/direct.h wavelets/innards.impl.h wavelets/wavelet_data.h
  wavelets/indirect.h wavelets/sara.h wavelets/wavelets.h
)

set(sources wavelets/wavelets.cc wavelets/wavelet_data.cc wavelets/sara.cc)

add_library(sopt SHARED ${sources})
add_dependencies(sopt lookup_dependencies)
set(version "${Sopt_VERSION_MAJOR}.${Sopt_VERSION_MINOR}.${Sopt_VERSION_PATCH}")
set_target_properties(sopt PROPERTIES VERSION ${version} SOVERSION ${version})
if(SPDLOG_INCLUDE_DIR)
  target_include_directories(sopt SYSTEM INTERFACE ${SPDLOG_INCLUDE_DIR})
endif()
if(EIGEN3_INCLUDE_DIR)
  target_include_directories(sopt SYSTEM INTERFACE ${EIGEN3_INCLUDE_DIR})
endif()

install(FILES ${headers} DESTINATION include/sopt)
install(FILES ${wavelet_headers} DESTINATION include/sopt/wavelets)
install(TARGETS sopt
  EXPORT SoptCPPTargets
  DESTINATION share/cmake/sopt
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include
)
