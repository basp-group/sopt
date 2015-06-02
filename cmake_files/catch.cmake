if(NOT tests)
    return()
endif()

# First finds or downloads catch
find_package(Catch)
if(NOT CATCH_FOUND)
  set(catch_url
      https://raw.githubusercontent.com/philsquared/Catch/develop/single_include/catch.hpp)
  set(catch_file "${EXTERNAL_ROOT}/include/catch.hpp")
  file(MAKE_DIRECTORY "${EXTERNAL_ROOT}/include")
  file(DOWNLOAD ${catch_url} "${catch_file}")
  file(SIZE "${catch_file}" size)
  if(size EQUAL 0)
    # CMake can't download over https if build lacks ssl. So use wget or curl
    find_package(Wget)
    if(WGET_FOUND)
      execute_process(COMMAND ${WGET_EXECUTABLE} ${catch_url} -O "${catch_file}")
    else()
      find_program(CURL_EXECUTABLE curl)
      execute_process(COMMAND ${CURL_EXECUTABLE} -L ${catch_url} -o "${catch_file}")
    endif()
  endif()
  find_package(Catch REQUIRED)
endif()

# Then adds a function to create a test
function(add_catch_test testname)
  cmake_parse_arguments(catch "" "WORKING_DIRECTORY" "LIBRARIES;LABELS" ${ARGN})

  # Source deduce from testname if possible
  unset(source)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${testname}.cc")
    set(source ${testname}.cc)
  endif()

  # Construct executable
  add_executable(test_${testname} ${source} ${catch_UNPARSED_ARGUMENTS})
  if(catch_LIBRARIES)
    target_link_libraries(${testname} ${catch_LIBRARIES})
  endif()
  include_directories(${CATCH_INCLUDE_DIR})

  unset(EXTRA_ARGS)
  if(catch_WORKING_DIRECTORY)
    set(EXTRA_ARGS WORKING_DIRECTORY ${catch_WORKING_DIRECTORY})
  endif()
  add_test(NAME ${testname} COMMAND test_${testname} ${EXTRA_ARGS})

  list(APPEND catch_LABELS catch)
  set_tests_properties(${testname} PROPERTIES LABELS "${catch_LABELS}")
endfunction()
