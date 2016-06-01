# list of headers
set(headers
  imaging_padmm.h logging.disabled.h maths.h proximal.h relative_variation.h sdmm.h
  wavelets.h conjugate_gradient.h l1_proximal.h logging.enabled.h padmm.h proximal_expression.h
  reweighted.h types.h wrapper.h exception.h linear_transform.h logging.h positive_quadrant.h
  real_type.h sampling.h utilities.h ${EXTERNAL_ROOT}/include/sopt/config.h
)
set(wavelet_headers
  wavelets/direct.h wavelets/indirect.h wavelets/innards.impl.h wavelets/sara.h
  wavelets/wavelet_data.h wavelets/wavelets.h
)

set(sources wavelets/wavelets.cc wavelets/wavelet_data.cc)

add_library(sopt SHARED ${sources})
add_dependencies(sopt lookup_dependencies)
set(version "${Sopt_VERSION_MAJOR}.${Sopt_VERSION_MINOR}.${Sopt_VERSION_PATCH}")
set_target_properties(sopt PROPERTIES VERSION ${version} SOVERSION ${version})

target_include_directories(sopt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include/>)
if(TIFF_FOUND)
  target_link_libraries(sopt ${TIFF_LIBRARY})
  target_include_directories(sopt SYSTEM PUBLIC ${TIFF_LIBRARY})
endif()
# Add spdlog as direct dependency if not downloaded
if(SPDLOG_INCLUDE_DIR AND NOT spdlog_BUILT_AS_EXTERNAL_PROJECT)
  target_include_directories(sopt SYSTEM PUBLIC ${SPDLOG_INCLUDE_DIR})
elseif(SPDLOG_INCLUDE_DIR AND spdlog_BUILT_AS_EXTERNAL_PROJECT)
  target_include_directories(sopt SYSTEM PUBLIC $<BUILD_INTERFACE:${SPDLOG_INCLUDE_DIR}>)
endif()
# Add eigen as direct dependency if not downloaded
if(EIGEN3_INCLUDE_DIR AND NOT Eigen3_BUILT_AS_EXTERNAL_PROJECT)
  target_include_directories(sopt SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
elseif(EIGEN3_INCLUDE_DIR AND Eigen3_BUILT_AS_EXTERNAL_PROJECT)
  target_include_directories(sopt SYSTEM PUBLIC $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>)
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
