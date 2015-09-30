#add_library(libwavlets CPP ${CMAKE_SOURCE_DIR}/cpp/wavelets)
include_directories( 
     ${PYTHON_INCLUDE_DIR}
     ${NUMPY_INCLUDE_DIRS}
     ${CMAKE_CURRENT_SOURCE_DIR}
     ${FFTW3_INCLUDE_DIR}
     ${CFitsIO_INCLUDE_DIR}
     ${TIFF_INCLUDE_DIR}
     ${CMAKE_SOURCE_DIR}/cpp
     ${CMAKE_SOURCE_DIR}/cpp/wavelets
     ${EIGEN3_INCLUDE_DIR})

 add_python_module(sopt CPP TARGETNAME pywavelets LIBRARIES wavelets FAKE_INIT wavelets.pyx)
