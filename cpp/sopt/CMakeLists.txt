# list of headers
set(headers
  # sopt headers proper
  conjugate_gradient.h logging.disabled.h proximal.h types.h exception.h logging.enabled.h
  sampling.h maths.h linear_transform.h sdmm.h wrapper.h logging.h
  relative_variation.h padmm.h l1_padmm.h ${PROJECT_BINARY_DIR}/include/sopt/config.h

  # wavelet headers
  wavelets/direct.h wavelets/innards.impl.h wavelets/wavelet_data.h
  wavelets/indirect.h wavelets/sara.h wavelets/wavelets.h
)

set(sources wavelets/wavelets.cc wavelets/wavelet_data.cc utilities.cc)

add_library(sopt SHARED ${sources})
add_dependencies(sopt lookup_dependencies)
set(version "${Sopt_VERSION_MAJOR}.${Sopt_VERSION_MINOR}.${Sopt_VERSION_PATCH}")
set_target_properties(sopt PROPERTIES VERSION ${version} SOVERSION ${version})
if(TIFF_FOUND)
  target_include_directories(sopt SYSTEM PUBLIC ${TIFF_INCLUDE_DIR})
  target_link_libraries(sopt ${TIFF_LIBRARY})
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
target_include_directories(sopt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include/sopt>
)

install(FILES ${headers} DESTINATION include/sopt)
install(TARGETS sopt
  EXPORT SoptCPPTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include/sopt
  DESTINATION share/cmake/sopt
  COMPONENT dev
)
