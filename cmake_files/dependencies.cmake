# Look for external software
find_package(FFTW3 COMPONENTS DOUBLE)
find_package(CBLAS REQUIRED)
set(SOPT_BLAS_H "${BLAS_INCLUDE_FILENAME}")
find_package(TIFF REQUIRED)

# On some (linux) machines we also need libm to compile sopt_demo*.c
# Make a half-hearted attempt at finding it.
# If it exists, it shouldn't be difficult.
find_library(M_LIBRARY m)

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
if(M_LIBRARY)
    list(APPEND DEPENDENCY_LIBRARIES "${M_LIBRARY}")
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")

if(FFTW3_DOUBLE_FOUND)
    add_definitions(-DSOPT_FFTW_INSTALLED)
endif()
