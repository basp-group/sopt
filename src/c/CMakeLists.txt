set(DEMO_SOURCES sopt_demo1.c sopt_demo1_padmm.c)
if(FFTW3_DOUBLE_FOUND)
    list(APPEND DEMO_SOURCES sopt_demo2.c sopt_demo2_padmm.c)
endif()
set(Sopt_SOURCES
    sopt_error.c  sopt_l2.c    sopt_ran.c        sopt_wavelet.c
    sopt_image.c  sopt_meas.c  sopt_sara.c       sopt_tv.c
    sopt_l1.c     sopt_prox.c  sopt_sparsemat.c  sopt_utility.c
)
set(ABOUT_SOURCE sopt_about.c)
set(TEST_SOURCE sopt_test.c)

include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/include")
include_directories(SYSTEM ${BLAS_INCLUDE_DIR} ${FFTW3_INCLUDE_DIR} ${TIFF_INCLUDE_DIR})
set(DEPENDENCY_LIBRARIES ${FFTW3_DOUBLE_LIBRARY} ${BLAS_LIBRARIES} ${TIFF_LIBRARY})

if(M_LIBRARY)
    list(APPEND DEPENDENCY_LIBRARIES "${M_LIBRARY}")
endif()
if(FFTW3_DOUBLE_FOUND)
    add_definitions(-DSOPT_FFTW_INSTALLED)
endif()

# Creates library
add_library(libsopt SHARED ${Sopt_SOURCES})
set_target_properties(libsopt PROPERTIES OUTPUT_NAME "sopt")
target_link_libraries(libsopt ${DEPENDENCY_LIBRARIES})
set_target_properties(libsopt PROPERTIES VERSION 1.0.0 SOVERSION 1.0.0)

# Add each demo
foreach(DEMO_SOURCE ${DEMO_SOURCES})
  # first extract name
  get_filename_component(DEMO_NAME ${DEMO_SOURCE} NAME_WE)
  # then creates executable
  add_executable(${DEMO_NAME} ${DEMO_SOURCE})
  target_link_libraries(${DEMO_NAME} libsopt)
endforeach()

# add test
if(tests AND FFTW3_DOUBLE_FOUND)
  add_executable(sopt_test ${TEST_SOURCE})
  target_link_libraries(sopt_test libsopt ${DEPENDENCY_LIBRARIES})
  add_test(
    NAME sopt
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/sopt_test
  )
endif()

# Add about
add_executable(sopt_about ${ABOUT_SOURCE})
set_target_properties(sopt_about PROPERTIES
  COMPILE_DEFINITIONS
     "SOPT_VERSION=\"${Sopt_VERSION}\";SOPT_BUILD=\"${Sopt_GITREF}\""
)
target_link_libraries(sopt_about libsopt)

# install library and about
install(TARGETS libsopt sopt_about
    EXPORT SoptTargets
    LIBRARY DESTINATION lib COMPONENT shlib
    ARCHIVE DESTINATION lib COMPONENT archive
    RUNTIME DESTINATION bin COMPONENT bin
)
