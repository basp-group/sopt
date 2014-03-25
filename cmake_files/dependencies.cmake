# Look for external software
find_package(FFTW3 COMPONENTS DOUBLE)
find_package(BLAS REQUIRED)
find_package(TIFF REQUIRED)

# find_package blas does not look for cblas.h
if(NOT EXISTS BLAS_INCLUDE_DIR)
  find_path(BLAS_INCLUDE_DIR cblas.h)
endif()

# Adds include directories
include_directories(
  ${BLAS_INCLUDE_DIR}
  ${FFTW3_INCLUDE_DIR}
  ${TIFF_INCLUDE_DIR}
)

# Target libraries for the different executables
set(DEPENDENCY_LIBRARIES
    ${FFTW3_DOUBLE_LIBRARY}
    ${BLAS_LIBRARIES}
    ${TIFF_LIBRARY}
)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")

if(FFTW3_DOUBLE_FOUND)
    add_definitions(-DSOPT_FFTW_INSTALLED)
endif()
