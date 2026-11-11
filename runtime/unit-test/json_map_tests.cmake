# JSON-only executable, available without libbpf or the broad runtime suite.
# No object loader, agent, syscall-server, or test BPF program dependencies.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    return()
endif()

if(NOT TARGET Catch2::Catch2WithMain)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)
endif()

add_executable(bpftime_json_map_tests
    ${CMAKE_CURRENT_LIST_DIR}/test_bpftime_json_map_compat.cpp)
target_link_libraries(bpftime_json_map_tests PRIVATE runtime Catch2::Catch2WithMain)
set_property(TARGET bpftime_json_map_tests PROPERTY CXX_STANDARD 20)
add_test(NAME bpftime_json_map_tests
    COMMAND bpftime_json_map_tests "[json-map-policy]" --warn UnmatchedTestSpec)
set(json_map_tests bpftime_json_map_tests)
if(BPFTIME_ENABLE_CUDA_ATTACH)
    # Separate exact-name filters fail if either CUDA-only case disappears.
    add_test(NAME bpftime_json_host_array_storage_tests
        COMMAND bpftime_json_map_tests "JSON host GPU arrays preserve host storage and metadata" --warn UnmatchedTestSpec)
    add_test(NAME bpftime_json_host_array_dimensions_tests
        COMMAND bpftime_json_map_tests "JSON host GPU arrays reject inconsistent dimensions" --warn UnmatchedTestSpec)
    list(APPEND json_map_tests
        bpftime_json_host_array_storage_tests bpftime_json_host_array_dimensions_tests)
endif()
set_tests_properties(${json_map_tests} PROPERTIES
    TIMEOUT 60
    ENVIRONMENT "BPFTIME_MAX_FD_COUNT=128;BPFTIME_SHM_MEMORY_MB=20")
