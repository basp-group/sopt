include_directories( 
     ${PYTHON_INCLUDE_DIR}
     ${NUMPY_INCLUDE_DIRS}
     ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_SOURCE_DIR}/cpp
     ${CMAKE_SOURCE_DIR}/cpp/wavelets
)
if(EIGEN3_INCLUDE_DIR)
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()

add_python_module(sopt CPP TARGETNAME pywavelets LIBRARIES wavelets FAKE_INIT wavelets.pyx)
