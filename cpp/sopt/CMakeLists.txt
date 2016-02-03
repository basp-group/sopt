# list of headers
set(headers
  conjugate_gradient.h logging.disabled.h proximal.h types.h exception.h logging.enabled.h
  sampling.h utility.h linear_transform.h sdmm.h wrapper.h ${EXTERNAL_ROOT}/include/sopt/logging.h
  relative_variation.h padmm.h l1_padmm.h)

configure_file(config.in.h "${EXTERNAL_ROOT}/include/sopt/config.h")

add_library(sopt INTERFACE)
if(SPDLOG_INCLUDE_DIR)
  set_target_properties(sopt PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${SPDLOG_INCLUDE_DIR})
endif()
set_target_properties(sopt PROPERTIES INTERFACE_DEPENDS lookup_dependencies)

install(FILES ${headers} DESTINATION include/sopt)
install(TARGETS sopt
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include/sopt
)
