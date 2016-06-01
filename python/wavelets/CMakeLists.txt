include_directories(SYSTEM ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_python_module(sopt CPP TARGETNAME pywavelets LIBRARIES sopt FAKE_INIT wavelets.pyx)
