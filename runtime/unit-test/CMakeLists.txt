if(${ENABLE_EBPF_VERIFIER} AND NOT TARGET Catch2)
    message(STATUS "Adding Catch2 by FetchContent from runtime/unit-test")
    Include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)

    # if not enable verifier, we will use the catch2 from submodule
endif()

find_package(Boost REQUIRED)

set(TEST_SOURCES
    maps/test_per_cpu_array.cpp
    maps/test_per_cpu_hash.cpp
    maps/test_shm_hash_maps.cpp
    maps/test_external_map_ops.cpp
    maps/test_bpftime_hash_map.cpp
    maps/kernel_unit_tests.cpp
    
    test_bpftime_shm_json.cpp
    attach_with_ebpf/test_attach_filter_with_ebpf.cpp
    attach_with_ebpf/test_attach_uprobe_with_ebpf.cpp
    attach_with_ebpf/test_helpers.cpp
    attach_with_ebpf/test_shm_progs_attach.cpp
    attach_with_ebpf/test_ufunc_register.cpp
    attach_with_ebpf/test_attach_replace.cpp

    tailcall/test_user_to_kernel_tailcall.cpp
)

option(TEST_LCOV "option for lcov" OFF)
add_executable(bpftime_runtime_tests ${TEST_SOURCES})

if (${TEST_LCOV}) 
    target_compile_options(bpftime_runtime_tests PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
endif()

set_property(TARGET bpftime_runtime_tests PROPERTY CXX_STANDARD 20)
add_dependencies(bpftime_runtime_tests runtime bpftime-object bpftime_frida_uprobe_attach_impl)

if (${TEST_LCOV}) 
    target_link_options(bpftime_runtime_tests PRIVATE -lgcov)
    target_link_libraries(bpftime_runtime_tests PRIVATE runtime bpftime-object Catch2::Catch2WithMain bpftime_frida_uprobe_attach_impl gcov)
else ()
    target_link_libraries(bpftime_runtime_tests PRIVATE runtime bpftime-object Catch2::Catch2WithMain bpftime_frida_uprobe_attach_impl)
endif()

target_include_directories(bpftime_runtime_tests PRIVATE ${BPFTIME_RUNTIME_INCLUDE} ${BPFTIME_OBJECT_INCLUDE_DIRS} ${Catch2_INCLUDE} ${Boost_INCLUDE} ${FRIDA_UPROBE_ATTACH_IMPL_INCLUDE})
add_test(NAME bpftime_runtime_tests COMMAND bpftime_runtime_tests)

# These are necessary ebpf program required by the test
set(used_ebpf_programs
    uprobe
    replace
    filter
    helpers
    ufunc
)

foreach(current ${used_ebpf_programs})
    set(curr_target_name bpftime_test_ebpf_prog_${current})
    add_ebpf_program_target(${curr_target_name} ${CMAKE_CURRENT_SOURCE_DIR}/assets/${current}.bpf.c ${CMAKE_CURRENT_BINARY_DIR}/${current}.bpf.o)
    add_dependencies(bpftime_runtime_tests ${curr_target_name})
    string(TOUPPER ${current} current_name_upper)
    target_compile_definitions(bpftime_runtime_tests PRIVATE EBPF_PROGRAM_PATH_${current_name_upper}=${CMAKE_CURRENT_BINARY_DIR}/${current}.bpf.o)
endforeach()
