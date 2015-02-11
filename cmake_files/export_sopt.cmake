# Exports Sopt so other packages can access it
export(TARGETS sopt_about libsopt
    FILE "${PROJECT_BINARY_DIR}/SoptTargets.cmake")

# Avoids creating an entry in the cmake registry.
if(NOT NOEXPORT)
    export(PACKAGE Sopt)
endif()

# First in binary dir
set(ALL_INCLUDE_DIRS 
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/include"
)
configure_File(cmake_files/SoptConfig.in.cmake
    "${PROJECT_BINARY_DIR}/SoptConfig.cmake" @ONLY
)
configure_File(cmake_files/SoptConfigVersion.in.cmake
    "${PROJECT_BINARY_DIR}/SoptConfigVersion.cmake" @ONLY
)

# Then for installation tree
set(ALL_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include/sopt")
configure_File(cmake_files/SoptConfig.in.cmake
    "${PROJECT_BINARY_DIR}/CMakeFiles/SoptConfig.cmake" @ONLY
)

# Finally install all files
install(FILES
    "${PROJECT_BINARY_DIR}/CMakeFiles/SoptConfig.cmake"
    "${PROJECT_BINARY_DIR}/SoptConfigVersion.cmake"
    DESTINATION share/cmake/sopt
    COMPONENT dev
)

install(EXPORT SoptTargets
    DESTINATION share/cmake/sopt
    COMPONENT dev
)

