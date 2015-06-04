cmake_minimum_required(VERSION 2.8)
project(Sopt CXX)

# Location of extra cmake includes for the project
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_files)
# Downloads and installs GreatCMakeCookOff
# It contains a number of cmake recipes
include(LookUp-GreatCMakeCookOff)

# Version and git hash id
include(VersionAndGitRef)
set_version(0.1)
get_gitref()

option(tests          "Enable testing"                         on)

# Add c++11 stuff
include(AddCPP11Flags)
include(CheckCXX11Features)
cxx11_feature_check(REQUIRED unique_ptr nullptr override constructor_delegate)

# search/install dependencies
include(dependencies)

# some common compilation flags
include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${EIGEN3_INCLUDE_DIR}"
)

add_subdirectory(wavelets)

if(tests)
  enable_testing()
  include(catch)
  add_subdirectory(tests)
endif()
