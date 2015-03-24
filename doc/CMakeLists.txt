if(DOXYGEN_FOUND)
    configure_file(Doxygen.in Doxyfile @ONLY)
    add_custom_target (doc
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
      SOURCES ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    )
endif()
