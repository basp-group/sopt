cmake_minimum_required(VERSION 2.8)
project(Sopt C)

# Location of extra cmake includes for the project
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_files)
# Downloads and installs GreatCMakeCookOff
# It contains a number of cmake recipes
include(LookUp-GreatCMakeCookOff)

# Version and git hash id
include(VersionAndGitRef)
set_version(0.1)
get_gitref()


# Add C99 flag, since code uses inline
# yeah, yeah it is GNU C, and others since for ever, but it was not part of the standard.
include(AddC99Flags)

option(tests          "Enable testing"                         on)

include(dependencies)
if(tests)
  enable_testing()
endif()

include_directories(
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/include"
)

# include exernal dependencies
add_subdirectory(src/c)
add_subdirectory(include)
add_subdirectory(doc)

# Exports all Sopt so other packages can access it
include(export_sopt)
