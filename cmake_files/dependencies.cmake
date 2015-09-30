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
  # Gets old sopt package directly from git and installs it locally
  # We will test against this old version
  lookup_package(Sopt
    REQUIRED DOWNLOAD_BY_DEFAULT
    PATHS "${EXTERNAL_ROOT}"
    NO_DEFAULT_PATH
    ARGUMENTS
      GIT_REPOSITORY "file:///${PROJECT_SOURCE_DIR}"
      GIT_TAG ${REGRESSION_ORACLE_ID}
  )
endif()
