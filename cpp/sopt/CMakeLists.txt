# list of headers
set(headers
  imaging_padmm.h logging.disabled.h maths.h proximal_expression.h conjugate_gradient.h
  l1_proximal.h logging.enabled.h padmm.h real_type.h exception.h linear_transform.h logging.h
  proximal.h relative_variation.h sampling.h utilities.h sdmm.h wavelets.h types.h wrapper.h
  ${PROJECT_BINARY_DIR}/include/sopt/config.h)
set(wavelet_headers
  wavelets/direct.h wavelets/indirect.h wavelets/innards.impl.h wavelets/sara.h
  wavelets/wavelet_data.h wavelets/wavelets.h)

set(sources wavelets/wavelets.cc wavelets/wavelet_data.cc utilities.cc)

add_library(sopt SHARED ${sources})
add_dependencies(sopt lookup_dependencies)
set(version "${Sopt_VERSION_MAJOR}.${Sopt_VERSION_MINOR}.${Sopt_VERSION_PATCH}")
set_target_properties(sopt PROPERTIES VERSION ${version} SOVERSION ${version})

set(public_headers
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include/sopt>)
if(TIFF_FOUND)
  list(APPEND public_headers ${TIFF_INCLUDE_DIR})
  target_link_libraries(sopt ${TIFF_LIBRARY})
endif()
# Add spdlog as direct dependency if not downloaded
if(SPDLOG_INCLUDE_DIR AND NOT spdlog_BUILT_AS_EXTERNAL_PROJECT)
  list(APPEND public_headers ${SPDLOG_INCLUDE_DIR})
elseif(SPDLOG_INCLUDE_DIR AND spdlog_BUILT_AS_EXTERNAL_PROJECT)
  list(APPEND public_headers $<BUILD_INTERFACE:${SPDLOG_INCLUDE_DIR}>)
endif()
# Add eigen as direct dependency if not downloaded
if(EIGEN3_INCLUDE_DIR AND NOT Eigen3_BUILT_AS_EXTERNAL_PROJECT)
  list(APPEND public_headers ${EIGEN3_INCLUDE_DIR})
elseif(EIGEN3_INCLUDE_DIR AND Eigen3_BUILT_AS_EXTERNAL_PROJECT)
  list(APPEND public_headers $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>)
endif()
target_include_directories(sopt PUBLIC ${public_headers})

install(FILES ${headers} DESTINATION include/sopt)
install(FILES ${wavelet_headers} DESTINATION include/sopt/wavelets)
install(TARGETS sopt
  EXPORT SoptCPPTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include/sopt
  )
