include(PackageLookup)  # check for existence, or install external projects

lookup_package(Eigen3 ARGUMENTS HG_REPOSITORY https://bitbucket.org/LukePratley/eigen)
if(logging)
  lookup_package(spdlog REQUIRED)
endif()

find_package(TIFF)
if(examples OR regression OR Csopt)
  if(NOT TIFF_FOUND)
    message(FATAL_ERROR "Examples, regression, and C library all require TIFF")
  endif()
endif()

if(regressions OR Csopt)
  find_package(FFTW3 REQUIRED DOUBLE)
endif()

if(regressions AND NOT Csopt)
  lookup_package(Sopt
    REQUIRED DOWNLOAD_BY_DEFAULT
    PATHS "${EXTERNAL_ROOT}"
    NO_DEFAULT_PATH
    KEEP
    ARGUMENTS
      GIT_REPOSITORY "git@github.com:astro-informatics/sopt"
      GIT_TAG ${REGRESSION_ORACLE_ID}
  )
elseif(regressions AND Csopt)
  set(Sopt_LIBRARIES libsopt)
  set(Sopt_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
endif()

if(openmp)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  else()
    message(STATUS "Could not find OpenMP. Compiling without.")
  endif()
endif()

if(Csopt)
  if(NOT archer)
    if(NOT DEFINED BLA_VENDOR AND NOT ENV{BLA_VENDOR} STREQUAL "")
      set(BLA_VENDOR $ENV{BLA_VENDOR})
    endif()
    if(NOT DEFINED BLA_VENDOR OR BLA_VENDOR STREQUAL "All" OR BLA_VENDOR MATCHES "Intel")
      cmake_policy(PUSH)
      cmake_policy(SET CMP0054 NEW)
      find_package(MKL QUIET)
      cmake_policy(POP)
      if(MKL_FOUND)
        set(BLAS_LIBRARIES ${MKL_LIBRARIES})
        set(BLAS_INCLUDE_DIR ${MKL_INCLUDE_DIRS})
      endif()
    endif()
    find_package(CBLAS REQUIRED)
  else()
    set(BLAS_INCLUDE_FILENAME "cblas.h")
  endif()
  # On some (linux) machines we also need libm to compile sopt_demo*.c
  # Make a half-hearted attempt at finding it.
  # If it exists, it shouldn't be difficult.
  find_library(M_LIBRARY m)

  find_package(Doxygen)
endif()
