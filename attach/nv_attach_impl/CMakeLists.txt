add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/syscall_id_list.h
    COMMAND /bin/bash ${CMAKE_CURRENT_SOURCE_DIR}/generate_syscall_id_table.sh "${CMAKE_CURRENT_BINARY_DIR}/syscall_id_list.h"
    USES_TERMINAL
)
add_custom_target(
    bpftime_syscall_trace_syscall_id_table
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/syscall_id_list.h
)

add_library(
    bpftime_syscall_trace_attach_impl STATIC
    src/syscall_trace_attach_impl.cpp
    src/syscall_table.cpp
    src/syscall_trace_attach_private_data.cpp
)

add_dependencies(bpftime_syscall_trace_attach_impl bpftime_base_attach_impl spdlog::spdlog bpftime_syscall_trace_syscall_id_table)
set(SYSCALL_TRACE_ATTACH_IMPL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE STRING "Include path of syscall trace impl")

target_include_directories(bpftime_syscall_trace_attach_impl PRIVATE ${SPDLOG_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR} PUBLIC ${BASE_ATTACH_IMPL_INCLUDE} ${SYSCALL_TRACE_ATTACH_IMPL_INCLUDE})

target_link_libraries(bpftime_syscall_trace_attach_impl PUBLIC bpftime_base_attach_impl spdlog::spdlog)

set_property(TARGET bpftime_syscall_trace_attach_impl PROPERTY CXX_STANDARD 20)

set(TEST_SOURCES
    test/test_private_data_parsing.cpp
    test/test_syscall_dispatch.cpp
)
option(TEST_LCOV "option for lcov" OFF)
add_executable(bpftime_syscall_trace_attach_tests ${TEST_SOURCES})

if (${TEST_LCOV}) 
    target_compile_options(bpftime_syscall_trace_attach_tests PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
endif()

if(${ENABLE_EBPF_VERIFIER} AND NOT TARGET Catch2)
    message(STATUS "Adding Catch2 by FetchContent at syscall_trace_attach_impl")
    Include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)

    # if not enable verifier, we will use the catch2 from submodule
endif()

add_dependencies(bpftime_syscall_trace_attach_tests Catch2 bpftime_syscall_trace_attach_impl spdlog::spdlog)
if (${TEST_LCOV}) 
    target_link_options(bpftime_syscall_trace_attach_tests PRIVATE -lgcov)
    target_link_libraries(bpftime_syscall_trace_attach_tests PRIVATE Catch2::Catch2WithMain bpftime_syscall_trace_attach_impl spdlog::spdlog gcov)
else ()
    target_link_libraries(bpftime_syscall_trace_attach_tests PRIVATE Catch2::Catch2WithMain bpftime_syscall_trace_attach_impl spdlog::spdlog)
endif()
target_include_directories(bpftime_syscall_trace_attach_tests PRIVATE ${SYSCALL_TRACE_ATTACH_IMPL_INCLUDE} ${Catch2_INCLUDE} ${SPDLOG_INCLUDE})

add_test(NAME bpftime_syscall_trace_attach_tests COMMAND bpftime_syscall_trace_attach_tests)

set_property(TARGET bpftime_syscall_trace_attach_tests PROPERTY CXX_STANDARD 20)
