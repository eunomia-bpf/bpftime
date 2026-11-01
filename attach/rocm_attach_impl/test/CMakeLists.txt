# A fake SDK lets the agent's real exported entry point be exercised without
# a GPU, including runtime/queue failures that are unsafe to induce on hardware.
add_library(bpftime_rocm_test_sdk SHARED mock_sdk.cpp)
set_target_properties(bpftime_rocm_test_sdk PROPERTIES OUTPUT_NAME rocprofiler-sdk
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mock-sdk")
target_include_directories(bpftime_rocm_test_sdk PRIVATE "${BPFTIME_ROCM_ROOT}/include")

add_executable(bpftime_rocm_profiler_tests profiler_test.cpp)
target_compile_features(bpftime_rocm_profiler_tests PRIVATE cxx_std_20)
target_compile_definitions(bpftime_rocm_profiler_tests PRIVATE AMD_INTERNAL_BUILD)
target_include_directories(bpftime_rocm_profiler_tests PRIVATE "${BPFTIME_ROCM_ROOT}/include")
target_link_libraries(bpftime_rocm_profiler_tests PRIVATE ${CMAKE_DL_LIBS})
add_dependencies(bpftime_rocm_profiler_tests bpftime-agent bpftime_rocm_test_sdk)
add_test(NAME rocm_profiler_lifecycle
    COMMAND ${CMAKE_COMMAND}
      "-DTEST=$<TARGET_FILE:bpftime_rocm_profiler_tests>"
      "-DAGENT=$<TARGET_FILE:bpftime-agent>"
      "-DMOCK_DIR=$<TARGET_FILE_DIR:bpftime_rocm_test_sdk>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_profiler.cmake")
set_tests_properties(rocm_profiler_lifecycle PROPERTIES TIMEOUT 30)

if(BPFTIME_ENABLE_FRIDA)
    add_test(NAME rocm_cli_launch
        COMMAND ${CMAKE_COMMAND} "-DCLI=$<TARGET_FILE:bpftime-cli-cpp>"
          "-DAGENT=$<TARGET_FILE:bpftime-agent>"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/check_cli.cmake")
    set_tests_properties(rocm_cli_launch PROPERTIES TIMEOUT 30)
endif()
