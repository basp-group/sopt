cmake_minimum_required(VERSION 2.8)
project(Sopt CXX)

# Location of extra cmake includes for the project
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_files)
# Downloads and installs GreatCMakeCookOff
# It contains a number of cmake recipes
include(LookUp-GreatCMakeCookOff)

# Version and git hash id
include(VersionAndGitRef)
set_version(2.0.0)
get_gitref()

option(tests          "Enable testing"                         on)
option(python         "Enable python"                          off)
option(benchmarks     "Enable benchmarking"                    off)
option(examples       "Enable Examples"                        on)
option(logging        "Enable logging"                         on)
option(regressions    "Enable regressions"                     off)
option(openmp         "Enable OpenMP"                          on)

if(regressions)
  enable_language(C)
endif()
if(tests)
  enable_testing()
endif()

# Add c++11 stuff
include(AddCPP11Flags)
include(CheckCXX11Features)
if(tests AND python)
  include(AddPyTest)
endif()
cxx11_feature_check(REQUIRED unique_ptr nullptr override constructor_delegate)
include(compilations)

# search/install dependencies
include(dependencies)
# sets rpath policy explicitly
if(CMAKE_VERSION VERSION_LESS 3.0)
    set_property(GLOBAL PROPERTY MACOSX_RPATH ON)
else()
    cmake_policy(SET CMP0042 NEW)
endif()

if(tests OR examples)
  enable_testing()
endif()

if(tests)
  include(AddCatchTest)
endif()
if(examples)
  include(AddExample)
endif()

if(benchmarks)
  include(AddBenchmark)
endif()
if(regressions)
  include(AddRegression)
endif()
if(python)
  include(python_dependencies)
  include(PythonModule)
endif()

if(python)
  add_subdirectory(python)
endif()

add_subdirectory(cpp)

# Exports all Sopt so other packages can access it
include(export_sopt)
