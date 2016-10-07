if(NOT TARGET examples)
  add_custom_target(examples COMMAND ctest -L examples ${PROJECT_BINARY_DIR})
endif()

function(add_example targetname)
  cmake_parse_arguments(example "NOTEST" "WORKING_DIRECTORY" "LIBRARIES;LABELS;DEPENDS" ${ARGN})

  # Source deduce from targetname if possible
  unset(source)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${targetname}.cc")
    set(source ${targetname}.cc)
  elseif("${example_UNPARSED_ARGUMENTS}" STREQUAL "")
    message(FATAL_ERROR "No source given or found for ${targetname}")
  endif()

  add_executable(example_${targetname} ${source} ${example_UNPARSED_ARGUMENTS})
  set_target_properties(example_${targetname} PROPERTIES OUTPUT_NAME ${targetname})

  if(example_LIBRARIES)
    target_link_libraries(example_${targetname} ${example_LIBRARIES})
  endif()
  add_dependencies(examples example_${targetname})
  if(example_DEPENDS)
    add_dependencies(example_${targetname} ${example_DEPENDS})
  endif()
  if(TARGET lookup_dependencies)
    add_dependencies(example_${targetname} lookup_dependencies)
  endif()

  # Add to tests
  if(NOT example_NOTEST)
    unset(EXTRA_ARGS)
    if(example_WORKING_DIRECTORY)
      set(EXTRA_ARGS WORKING_DIRECTORY ${example_WORKING_DIRECTORY})
    endif()
    add_test(NAME test_example_${targetname} COMMAND example_${targetname} ${EXTRA_ARGS})

    list(APPEND example_LABELS examples)
    set_tests_properties(test_example_${targetname} PROPERTIES LABELS "${example_LABELS}")
  endif()
endfunction()
