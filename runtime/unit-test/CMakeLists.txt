Include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.0.1
)

FetchContent_MakeAvailable(Catch2)

find_package(Boost REQUIRED)

set(TEST_SOURCES
    test_per_cpu_array.cpp
    test_per_cpu_hash.cpp
    attach_manager/test_uprobe_uretprobe.cpp
    attach_manager/test_function_address_resolve.cpp
    attach_manager/test_filter_attach.cpp
    attach_manager/test_replace_attach.cpp
)

add_executable(bpftime_runtime_tests ${TEST_SOURCES})
set_property(TARGET bpftime_runtime_tests PROPERTY CXX_STANDARD 20)
add_dependencies(bpftime_runtime_tests runtime)
target_link_libraries(bpftime_runtime_tests PRIVATE runtime Catch2::Catch2WithMain)
target_include_directories(bpftime_runtime_tests PRIVATE ${BPFTIME_RUNTIME_INCLUDE} ${Catch2_INCLUDE} ${Boost_INCLUDE})
add_test(NAME bpftime_runtime_tests COMMAND bpftime_runtime_tests)

# These are necessary ebpf program required by the test
set(used_ebpf_programs
    uprobe
    replace
    filter)

foreach(current ${used_ebpf_programs})
    set(curr_target_name bpftime_test_ebpf_prog_${current})
    add_ebpf_program_target(${curr_target_name} ${CMAKE_CURRENT_SOURCE_DIR}/assets/${current}.bpf.c ${CMAKE_CURRENT_BINARY_DIR}/${current}.bpf.o)
    add_dependencies(bpftime_runtime_tests ${curr_target_name})
    string(TOUPPER current_name_upper ${current})
    target_compile_definitions(bpftime_runtime_tests PRIVATE EBPF_PROGRAM_PATH_${current_name_upper}=${CMAKE_CURRENT_BINARY_DIR}/${current}.bpf.o)
endforeach()
