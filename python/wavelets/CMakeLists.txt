include_directories(SYSTEM ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/cpp)
if(EIGEN3_INCLUDE_DIR)
  include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
endif()

add_python_module(sopt CPP TARGETNAME pywavelets LIBRARIES sopt FAKE_INIT wavelets.pyx)
