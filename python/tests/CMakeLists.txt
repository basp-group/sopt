#if(tests AND python)
message("${PYTHON_EXECUTABLE}")
setup_pytest("${EXTERNAL_ROOT}/python" "${PROJECT_BINARY_DIR}/py.test.sh")

add_pytest(test_*.py)
      #     sparse_testing.pyx visibility_testing.pyx random.pyx
      #  image_fixtures.py
      #  DEPENDS copy
      #  FAKE_INIT
      #  PREFIX purify
      #  LIBRARIES libpurify ${Sopt_LIBRARIES}
      #endif()
