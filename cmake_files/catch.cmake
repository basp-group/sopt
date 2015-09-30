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
  file(READ "${catch_file}" CATCHSTRING LIMIT 1000)
  string(LENGTH "${CATCHSTRING}" CATCHLENGTH)
  if(NOT CATCHLENGTH GREATER 500)
    # CMake can't download over https if build lacks ssl. So use wget or curl
    find_package(Wget)
    if(WGET_FOUND)
      execute_process(COMMAND ${WGET_EXECUTABLE} ${catch_url} -O "${catch_file}")
    else()
      find_program(CURL_EXECUTABLE curl)
      execute_process(COMMAND ${CURL_EXECUTABLE} -L ${catch_url} -o "${catch_file}")
    endif()
    file(READ "${catch_file}" CATCHSTRING LIMIT 1000)
    string(LENGTH "${CATCHSTRING}" CATCHLENGTH)
    if(NOT CATCHLENGTH GREATER 500)
      file(REMOVE "${catch_file}")
      message(FATAL_ERROR "Failed to download Catch ${CATCHSTRING} ${CATCHLENGTH}")
    endif()
  endif()
  find_package(Catch REQUIRED)
endif()

# Function to create a common main
function(common_catch_main)
  if(TARGET common_catch_main_object)
    return()
  endif()
  file(WRITE "${CMAKE_BINARY_DIR}/common_catch_main.cc"
    "#define CATCH_CONFIG_MAIN\n"
    "#include \"catch.hpp\"\n"
  )
  add_library(common_catch_main_object OBJECT "${CMAKE_BINARY_DIR}/common_catch_main.cc")
  if(CATCH_INCLUDE_DIR)
    target_include_directories(common_catch_main_object PRIVATE ${CATCH_INCLUDE_DIR})
  endif()
endfunction()

# Then adds a function to create a test
function(add_catch_test testname)
  cmake_parse_arguments(catch
    "NOMAIN;NOTEST;NOCATCHLABEL"
    "WORKING_DIRECTORY;SEED"
    "LIBRARIES;LABELS;DEPENDS;ARGUMENTS;INCLUDES"
    ${ARGN}
  )

  # Source deduce from testname if possible
  unset(source)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${testname}.cc")
    set(source ${testname}.cc)
  elseif("${catch_UNPARSED_ARGUMENTS}" STREQUAL "")
    message(FATAL_ERROR "No source given or found for ${testname}")
  endif()


  # By default, uses a common main function for all, compiled once
  # We create here
  if(catch_NOMAIN)
    add_executable(test_${testname} ${source} ${catch_UNPARSED_ARGUMENTS})
  else()
    common_catch_main()
    # Construct executable
    add_executable(test_${testname}
      ${source} $<TARGET_OBJECTS:common_catch_main_object> ${catch_UNPARSED_ARGUMENTS})
  endif()

  if(catch_LIBRARIES)
    target_link_libraries(test_${testname} ${catch_LIBRARIES})
  endif()
  if(CATCH_INCLUDE_DIR)
    target_include_directories(test_${testname} PRIVATE ${CATCH_INCLUDE_DIR})
  endif()
  if(catch_DEPENDS)
    add_dependencies(test_${testname} ${catch_DEPENDS})
  endif()

  unset(EXTRA_ARGS)
  if(catch_WORKING_DIRECTORY)
    set(EXTRA_ARGS WORKING_DIRECTORY ${catch_WORKING_DIRECTORY})
  endif()
  set(arguments ${catch_ARGUMENTS})
  if(catch_SEED)
    list(APPEND arguments --rng-seed ${catch_SEED})
  else()
    list(APPEND arguments --rng-seed time)
  endif()
  if(NOT catch_NOTEST)
    if(CATCH_JUNIT)
      add_test(NAME ${testname}
        COMMAND test_${testname}
            ${arguments}
            -r junit
            -o ${PROJECT_BINARY_DIR}/Testing/${testname}.xml
      )
    else()
      add_test(NAME ${testname} COMMAND test_${testname} ${arguments} ${EXTRA_ARGS})
    endif()

    if(NOT catch_NOCATCHLABEL)
      list(APPEND catch_LABELS catch)
    endif()
    set_tests_properties(${testname} PROPERTIES LABELS "${catch_LABELS}")
  endif()
endfunction()
