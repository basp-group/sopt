include(PackageLookup)  # check for existence, or install external projects

lookup_package(Eigen3 REQUIRED)
if(logging)
  lookup_package(spdlog REQUIRED)
endif()

if(examples OR regression)
  find_package(TIFF REQUIRED)
endif()
if(regressions)
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
