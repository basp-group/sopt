if(NOT TARGET regressions)
  add_custom_target(regressions COMMAND ctest -L regressions ${PROJECT_BINARY_DIR})
endif()

include(AddCatchTest)
function(add_regression targetname)
  cmake_parse_arguments(regr "" "" "LIBRARIES;LABELS;INCLUDES" ${ARGN})
  add_catch_test(${targetname} ${regr_UNPARSED_ARGUMENTS}
    LIBRARIES ${Sopt_LIBRARIES} ${regr_LIBRARIES}
    LABELS ${regr_LABELS} "regression"
    INCLUDES ${regr_INCLUDE} ${TIFF_INCLUDE_DIR} ${Sopt_INCLUDE_DIRS}
  )
endfunction()
