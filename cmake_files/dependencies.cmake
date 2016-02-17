include(PackageLookup)  # check for existence, or install external projects

lookup_package(Eigen3 REQUIRED)
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

if(Csopt)
  find_package(CBLAS REQUIRED)
  # On some (linux) machines we also need libm to compile sopt_demo*.c
  # Make a half-hearted attempt at finding it.
  # If it exists, it shouldn't be difficult.
  find_library(M_LIBRARY m)

  find_package(Doxygen)
endif()
