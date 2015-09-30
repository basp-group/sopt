# Looks up [Sopt](http://basp-group.github.io/sopt/)
#
# - GIT_REPOSITORY: defaults to https://github.com/UCL/sopt.git
# - GIT_TAG: defaults to master
# - BUILD_TYPE: defaults to Release
#
if(Sopt_ARGUMENTS)
    cmake_parse_arguments(Sopt "" "GIT_REPOSITORY;GIT_TAG;BUILD_TYPE" ""
        ${Sopt_ARGUMENTS})
endif()
if(NOT Sopt_GIT_REPOSITORY)
    set(Sopt_GIT_REPOSITORY https://github.com/UCL/sopt.git)
endif()
if(NOT Sopt_GIT_TAG)
    set(Sopt_GIT_TAG master)
endif()
if(NOT Sopt_BUILD_TYPE)
    set(Sopt_BUILD_TYPE Release)
endif()

# write subset of variables to cache for sopt to use
include(PassonVariables)
passon_variables(Sopt
  FILENAME "${EXTERNAL_ROOT}/src/SoptVariables.cmake"
  PATTERNS
      "CMAKE_[^_]*_R?PATH" "CMAKE_C_.*"
      "BLAS_.*" "FFTW3_.*" "TIFF_.*"
      "GreatCMakeCookOff_DIR"
  ALSOADD
      "\nset(CMAKE_INSTALL_PREFIX \"${EXTERNAL_ROOT}\" CACHE STRING \"\")\n"
)
ExternalProject_Add(
    Sopt
    PREFIX ${EXTERNAL_ROOT}
    GIT_REPOSITORY ${Sopt_GIT_REPOSITORY}
    GIT_TAG ${Sopt_GIT_TAG}
    CMAKE_ARGS
      -C "${EXTERNAL_ROOT}/src/SoptVariables.cmake"
      -DBUILD_SHARED_LIBS=OFF
      -DCMAKE_BUILD_TYPE=${Sopt_BUILD_TYPE}
      -DNOEXPORT=TRUE
    INSTALL_DIR ${EXTERNAL_ROOT}
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
)
add_recursive_cmake_step(Sopt DEPENDEES install)
