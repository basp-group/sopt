# Scripts to run purify from build directory. Good for testing/debuggin.
include(EnvironmentScript)
# Function to install python files in python path ${PYTHON_PKG_DIR}
include(PythonInstall)
# Ability to find installed python packages
include(PythonPackage)
# Installs python packages that are missing.
# We choose to do this only for packages that are required to build purify.
# Leave it to the user to install packages that are needed for running
# purify.
include(PythonPackageLookup)
# Creates script for running python with the bempp package available
# Also makes python packages and selected directories available to the
# build system
add_to_python_path("${PROJECT_BINARY_DIR}/python")
add_to_python_path("${EXTERNAL_ROOT}/python")
add_python_eggs("${PROJECT_SOURCE_DIR}"
    EXCLUDE
        "${PROJECT_SOURCE_DIR}/purify*egg"
        "${PROJECT_BINARY_DIR}/Purify*egg"
)
set(LOCAL_PYTHON_EXECUTABLE "${PROJECT_BINARY_DIR}/localpython.sh")
create_environment_script(
    EXECUTABLE "${PYTHON_EXECUTABLE}"
    PATH "${LOCAL_PYTHON_EXECUTABLE}"
    PYTHON
)
# Python interpreter + libraries
find_package(CoherentPython)
# Only required for building
if(NOT cython_EXECUTABLE)
    lookup_python_package(Cython REQUIRED PATH "${EXTERNAL_ROOT}/python")
endif()
# Also required for production
find_python_package(numpy)
find_python_package(scipy)
find_python_package(pandas)
# Finds additional info, like libraries, include dirs...
# no need check numpy features, it's all handled by cython.
set(no_numpy_feature_tests TRUE)
find_package(Numpy REQUIRED)

if(tests)
    include(AddPyTest)
    setup_pytest("${EXTERNAL_ROOT}/python" "${PROJECT_BINARY_DIR}/py.test.sh")
endif()

