get_filename_component(Sopt_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" PATH)
set(Sopt_INCLUDE_DIRS "@ALL_INCLUDE_DIR@")
if(NOT TARGET libsopt and NOT Sopt_BINARY_DIR)
    include("${Sopt_CMAKE_DIR}/SoptTargets.cmake")
endif()

set(Sopt_LIBRARIES libsopt)
set(Sopt_ABOUT_EXECUTABLE sopt_about)
