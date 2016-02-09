include(PackageLookup)  # check for existence, or install external projects

lookup_package(Eigen3 REQUIRED)
if(logging)
  lookup_package(spdlog REQUIRED)
endif()

if(examples OR regression OR Csopt)
  find_package(TIFF REQUIRED)
endif()
if(regressions AND NOT Csopt)
  find_package(FFTW3 REQUIRED DOUBLE)
  lookup_package(Sopt
    REQUIRED DOWNLOAD_BY_DEFAULT
    PATHS "${EXTERNAL_ROOT}"
    NO_DEFAULT_PATH
    KEEP
    ARGUMENTS
      GIT_REPOSITORY "git@github.com:astro-informatics/sopt"
      GIT_TAG ${REGRESSION_ORACLE_ID}
  )
endif()

if(Csopt)
  find_package(FFTW3 COMPONENTS DOUBLE)
  find_package(CBLAS REQUIRED)
  # On some (linux) machines we also need libm to compile sopt_demo*.c
  # Make a half-hearted attempt at finding it.
  # If it exists, it shouldn't be difficult.
  find_library(M_LIBRARY m)

  find_package(Doxygen)
endif()
