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
    maps/test_fix_hash_map.cpp
    maps/test_map_in_map_bounds.cpp
    maps/kernel_unit_tests.cpp
    maps/test_stack_trace_map.cpp
    maps/test_queue_map.cpp
    maps/test_stack_map.cpp
    maps/test_bloom_filter_map.cpp
    maps/test_lpm_trie_map.cpp
    maps/test_ringbuf_map.cpp
    maps/test_ringbuf_reserve.cpp
    maps/test_lpm_trie_inline_value.cpp
    maps/test_map_lock_dead_owner.cpp

    maps/test_lru_var_hash_map.cpp
    maps/test_lru_hash_map.cpp
    maps/test_map_flags_enforcement.cpp
    maps/test_map_helper_errno.cpp
    maps/test_map_syscall_semantics.cpp
    maps/test_array_get_next_key.cpp
    maps/test_map_ops_fuzzer_replay.cpp
    # The libFuzzer entry point, replayed by test_map_ops_fuzzer_replay.cpp.
    ../fuzz/map_ops_fuzzer.cpp

    test_bpftime_shm_json.cpp
    test_probe.cpp
    test_process_filter.cpp
    test_cpu_execution.cpp
    test_get_current_pid_tgid.cpp
    test_get_stack.cpp
    test_tracepoint_id.cpp
    test_config.cpp
    test_shm_allocation_failure.cpp
    test_atomic_pid_set.cpp
    test_software_perf_event.cpp
    test_software_perf_event_cache.cpp
    test_ringbuf_tail_wrap.cpp
    test_epoll.cpp
    test_unregistered_helper.cpp
    test_xdp_adjust_head.cpp
    test_synthetic_kallsyms.cpp
    test_mocked_file_fd.cpp
    test_system_analysis_ptregs.cpp
    test_handle_fd_reservation.cpp
    test_attach_failures.cpp
    test_memfd_create.cpp
    test_robust_shm_mutex.cpp

    attach_with_ebpf/test_attach_uprobe_with_ebpf_trap.cpp
    attach_with_ebpf/test_helpers.cpp
    attach_with_ebpf/test_ufunc_register.cpp

    tailcall/test_user_to_kernel_tailcall.cpp
    tailcall/test_user_to_user_tailcall.cpp
    tailcall/test_prog_array_thread_local.cpp

)
# These tests instantiate attach::frida_attach_impl directly, so they only
# build when the frida backend is available (it is off on riscv64).
if(BPFTIME_ENABLE_FRIDA)
    list(APPEND TEST_SOURCES
        attach_with_ebpf/test_attach_filter_with_ebpf.cpp
        attach_with_ebpf/test_attach_uprobe_with_ebpf.cpp
        attach_with_ebpf/test_shm_progs_attach.cpp
        attach_with_ebpf/test_attach_replace.cpp
    )
endif()

if(${BPFTIME_ENABLE_CUDA_ATTACH})
    list(APPEND TEST_SOURCES cuda/test_cuda_compile.cpp)
endif()
option(TEST_LCOV "option for lcov" OFF)
add_executable(bpftime_runtime_tests ${TEST_SOURCES})

# Compile the header fallback without CMake's DEFAULT_VM_NAME override.
# Keep this object unlinked so its runtime_config definition cannot violate ODR.
add_library(bpftime_config_header_test OBJECT test_default_vm_header.cpp)
target_include_directories(bpftime_config_header_test PRIVATE
    ${BPFTIME_RUNTIME_INCLUDE} ${Boost_INCLUDE_DIRS})
target_compile_definitions(bpftime_config_header_test PRIVATE
    $<$<BOOL:${BPFTIME_LLVM_JIT}>:BPFTIME_LLVM_JIT=1>
    $<$<BOOL:${BPFTIME_UBPF_JIT}>:BPFTIME_UBPF_JIT=1>)
set_property(TARGET bpftime_config_header_test PROPERTY CXX_STANDARD 20)
add_dependencies(bpftime_runtime_tests bpftime_config_header_test)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_sources(bpftime_runtime_tests PRIVATE
        ../syscall-server/syscall_server_utils.cpp
        ../syscall-server/syscall_context.cpp
        ../syscall-server/rootless_map_files.cpp
        ../syscall-server/seccomp_raw_syscall.cpp)
    target_include_directories(bpftime_runtime_tests PRIVATE ../syscall-server)
    target_link_libraries(bpftime_runtime_tests PRIVATE bpftime_syscall_trace_attach_impl ${CMAKE_DL_LIBS})
    target_link_options(bpftime_runtime_tests PRIVATE -Wl,--wrap=dlsym)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        target_sources(bpftime_runtime_tests PRIVATE
            test_rootless_seccomp_constructor.cpp
            syscall_server/test_seccomp_fastpath.cpp)
    endif()
    target_sources(bpftime_runtime_tests PRIVATE
        test_get_current_uid_gid.cpp
        test_get_current_comm.cpp)
    target_link_options(bpftime_runtime_tests PRIVATE
        -Wl,--wrap=getuid -Wl,--wrap=getgid
        # Runtime unit tests require BPFTIME_BUILD_WITH_LIBBPF.
        -Wl,--wrap=btf__parse_raw
        -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc)
endif()

if(ENABLE_PROBE_READ_CHECK)
    target_compile_definitions(bpftime_runtime_tests PRIVATE ENABLE_PROBE_READ_CHECK=1)
endif()

if(ENABLE_PROBE_WRITE_CHECK)
    target_compile_definitions(bpftime_runtime_tests PRIVATE ENABLE_PROBE_WRITE_CHECK=1)
endif()

if(UNIX AND NOT APPLE)
    add_executable(bpftime-shm-allocation-test-helper
        assets/shm_allocation_test_helper.c)
    add_executable(bpftime-runtime-options-test-helper
        assets/runtime_options_test_helper.c)
    add_executable(bpftime-map-syscall-test-helper
        assets/map_syscall_test_helper.c)
    set_property(TARGET bpftime-shm-allocation-test-helper PROPERTY C_STANDARD 99)
    set_property(TARGET bpftime-runtime-options-test-helper PROPERTY C_STANDARD 99)
    set_property(TARGET bpftime-map-syscall-test-helper PROPERTY C_STANDARD 99)
    # The rootless constructor tests need a process that bypasses glibc
    # __libc_start_main (custom _start, raw syscalls only) and an ordinary
    # glibc process built from the same helper source. x86_64-only: the
    # rootless seccomp USER_NOTIF entry itself is x86_64-only.
    set(_rootless_custom_helper_expr "")
    set(_rootless_normal_helper_expr "")
    set(_seccomp_fastpath_helper_expr "")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        add_executable(bpftime-rootless-custom-start-helper
            assets/rootless_start_helper.c)
        target_compile_definitions(bpftime-rootless-custom-start-helper
            PRIVATE BPFTIME_CUSTOM_START=1)
        target_link_options(bpftime-rootless-custom-start-helper
            PRIVATE -nostartfiles)
        add_executable(bpftime-rootless-normal-start-helper
            assets/rootless_start_helper.c)
        set_property(TARGET bpftime-rootless-custom-start-helper
                                  bpftime-rootless-normal-start-helper
                     PROPERTY C_STANDARD 99)
        set(_rootless_custom_helper_expr
            "$<TARGET_FILE:bpftime-rootless-custom-start-helper>")
        set(_rootless_normal_helper_expr
            "$<TARGET_FILE:bpftime-rootless-normal-start-helper>")
        add_dependencies(bpftime_runtime_tests
            bpftime-rootless-custom-start-helper
            bpftime-rootless-normal-start-helper)
        add_executable(bpftime-seccomp-fastpath-test-helper
            assets/seccomp_fastpath_helper.c)
        set_property(TARGET bpftime-seccomp-fastpath-test-helper
                     PROPERTY C_STANDARD 11)
        add_dependencies(bpftime_runtime_tests
            bpftime-seccomp-fastpath-test-helper)
        set(_seccomp_fastpath_helper_expr
            "$<TARGET_FILE:bpftime-seccomp-fastpath-test-helper>")
    endif()
    add_dependencies(bpftime_runtime_tests
        bpftime-agent
        bpftime-syscall-server
        bpftime-runtime-options-test-helper
        bpftime-map-syscall-test-helper
        bpftime-shm-allocation-test-helper)
    file(GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shm_allocation_test_paths.hpp"
        CONTENT "#define BPFTIME_SHM_ALLOCATION_TEST_HELPER \"$<TARGET_FILE:bpftime-shm-allocation-test-helper>\"\n#define BPFTIME_MAP_SYSCALL_TEST_HELPER \"$<TARGET_FILE:bpftime-map-syscall-test-helper>\"\n#define BPFTIME_SYSCALL_SERVER_LIBRARY \"$<TARGET_FILE:bpftime-syscall-server>\"\n#define BPFTIME_AGENT_LIBRARY \"$<TARGET_FILE:bpftime-agent>\"\n#define BPFTIME_ROOTLESS_CUSTOM_START_HELPER \"${_rootless_custom_helper_expr}\"\n#define BPFTIME_ROOTLESS_NORMAL_START_HELPER \"${_rootless_normal_helper_expr}\"\n#define BPFTIME_SECCOMP_FASTPATH_TEST_HELPER \"${_seccomp_fastpath_helper_expr}\"\n")
    target_include_directories(bpftime_runtime_tests PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}")
endif()

if(${TEST_LCOV})
    target_compile_options(bpftime_runtime_tests PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
endif()

set_property(TARGET bpftime_runtime_tests PROPERTY CXX_STANDARD 20)
if(UNIX AND NOT APPLE AND BPFTIME_LLVM_JIT)
    # ORC resolves its statically linked EH-frame registration wrappers via
    # the process symbol table (also needed by existing x86 runtime tests).
    target_link_options(bpftime_runtime_tests PRIVATE -Wl,--export-dynamic)
endif()
add_dependencies(bpftime_runtime_tests runtime bpftime-object)
if(TARGET bpftime_frida_uprobe_attach_impl)
    add_dependencies(bpftime_runtime_tests bpftime_frida_uprobe_attach_impl)
    target_link_libraries(bpftime_runtime_tests PRIVATE bpftime_frida_uprobe_attach_impl)
    target_include_directories(bpftime_runtime_tests PRIVATE ${FRIDA_UPROBE_ATTACH_IMPL_INCLUDE})
endif()
if(TARGET bpftime_trap_uprobe_attach_impl)
    target_link_libraries(bpftime_runtime_tests PRIVATE bpftime_trap_uprobe_attach_impl)
    target_include_directories(bpftime_runtime_tests PRIVATE ${TRAP_UPROBE_ATTACH_IMPL_INCLUDE})
    target_compile_definitions(bpftime_runtime_tests PRIVATE BPFTIME_HAVE_TRAP_UPROBE_ATTACH=1)
endif()

if(${BPFTIME_ENABLE_CUDA_ATTACH})
    include(../../cmake/cuda.cmake)
    find_cuda()
    target_include_directories(bpftime_runtime_tests PRIVATE ${CUDA_INCLUDE_PATH})
    target_link_directories(bpftime_runtime_tests PRIVATE ${CUDA_LIBRARY_PATH})

endif()

if(${TEST_LCOV})
    target_link_options(bpftime_runtime_tests PRIVATE -lgcov)
    target_link_libraries(bpftime_runtime_tests PRIVATE runtime bpftime-object Catch2::Catch2WithMain gcov ${CUDA_LIBS})
else()
    target_link_libraries(bpftime_runtime_tests PRIVATE runtime bpftime-object Catch2::Catch2WithMain ${CUDA_LIBS})
endif()

target_include_directories(bpftime_runtime_tests PRIVATE ${BPFTIME_RUNTIME_INCLUDE} ${BPFTIME_OBJECT_INCLUDE_DIRS} ${Catch2_INCLUDE} ${Boost_INCLUDE} "${CMAKE_CURRENT_SOURCE_DIR}/../syscall-server")
add_test(NAME bpftime_runtime_tests COMMAND bpftime_runtime_tests)

# These are necessary ebpf program required by the test
set(used_ebpf_programs
    uprobe
    uprobe_trap
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
