find_package(Boost REQUIRED)

set(TEST_SOURCES
    test_daemon.cpp
    test_daemon_driver.cpp
)

add_executable(bpftime_daemon_tests ${TEST_SOURCES})
set_property(TARGET bpftime_daemon_tests PROPERTY CXX_STANDARD 20)
add_dependencies(bpftime_daemon_tests runtime bpftime-object)
target_link_libraries(bpftime_daemon_tests PRIVATE
    runtime
    bpftime-object
    Catch2::Catch2WithMain
    libbpftime_daemon
)
target_include_directories(bpftime_daemon_tests
    PRIVATE
    ${BPFTIME_RUNTIME_INCLUDE}
    ${BPFTIME_OBJECT_INCLUDE_DIRS}
    ${Catch2_INCLUDE}
    ${Boost_INCLUDE}
    ../
)
add_test(NAME bpftime_daemon_tests COMMAND bpftime_daemon_tests)

# These are necessary ebpf program required by the test
# set(used_ebpf_programs
# uprobe
# # replace # Not used now
# filter)
foreach(current ${used_ebpf_programs})
    set(curr_target_name bpftime_test_ebpf_prog_${current})
    add_ebpf_program_target(${curr_target_name} ${CMAKE_CURRENT_SOURCE_DIR}/assets/${current}.bpf.c ${CMAKE_CURRENT_BINARY_DIR}/${current}.bpf.o)
    add_dependencies(bpftime_runtime_tests ${curr_target_name})
    string(TOUPPER ${current} current_name_upper)
    target_compile_definitions(bpftime_runtime_tests PRIVATE EBPF_PROGRAM_PATH_${current_name_upper}=${CMAKE_CURRENT_BINARY_DIR}/${current}.bpf.o)
endforeach()
