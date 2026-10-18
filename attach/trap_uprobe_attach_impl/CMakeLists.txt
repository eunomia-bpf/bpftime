add_library(
    bpftime_trap_uprobe_attach_impl STATIC
    src/trap_uprobe_attach_impl.cpp
    src/trap_attach_private_data.cpp
    src/trap_attach_utils.cpp
    src/trap_arch_riscv64.cpp
)
add_dependencies(bpftime_trap_uprobe_attach_impl bpftime_base_attach_impl spdlog::spdlog)

set(TRAP_UPROBE_ATTACH_IMPL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE STRING "Include path of trap uprobe impl")

target_include_directories(bpftime_trap_uprobe_attach_impl
    PRIVATE ${SPDLOG_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC ${TRAP_UPROBE_ATTACH_IMPL_INCLUDE} ${BASE_ATTACH_IMPL_INCLUDE})

target_link_libraries(bpftime_trap_uprobe_attach_impl PUBLIC bpftime_base_attach_impl spdlog::spdlog dl pthread)
# Resolve PLT entries before the first signal, including in late-loaded DSOs.
target_link_options(bpftime_trap_uprobe_attach_impl INTERFACE -Wl,-z,now)

set_property(TARGET bpftime_trap_uprobe_attach_impl PROPERTY CXX_STANDARD 20)
set_property(TARGET bpftime_trap_uprobe_attach_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

if(BPFTIME_TRAP_ATTACH_BUILD_TESTS)
    set(TRAP_TEST_SOURCES
        test/test_trap_uprobe_uretprobe.cpp
        test/test_trap_override.cpp
        test/test_trap_unified_interface.cpp
        test/test_trap_misc.cpp
        test/test_trap_multithread.cpp
        test/test_trap_backtrace.cpp
        test/test_trap_recursion.cpp
        test/test_trap_insn_decode.cpp
        test/test_trap_signal_safety.cpp
        test/test_trap_concurrent_patch.cpp
        test/test_trap_patch_failure.cpp
    )
    add_executable(bpftime_trap_uprobe_attach_tests ${TRAP_TEST_SOURCES})
    add_dependencies(bpftime_trap_uprobe_attach_tests Catch2 bpftime_trap_uprobe_attach_impl spdlog::spdlog)
    target_link_libraries(bpftime_trap_uprobe_attach_tests PRIVATE Catch2::Catch2WithMain bpftime_trap_uprobe_attach_impl spdlog::spdlog pthread)
    target_link_options(bpftime_trap_uprobe_attach_tests PRIVATE -Wl,--wrap=mprotect)
    target_include_directories(bpftime_trap_uprobe_attach_tests PRIVATE ${TRAP_UPROBE_ATTACH_IMPL_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/src ${Catch2_INCLUDE} ${SPDLOG_INCLUDE})
    set_property(TARGET bpftime_trap_uprobe_attach_tests PROPERTY CXX_STANDARD 20)
    add_test(NAME bpftime_trap_uprobe_attach_tests COMMAND bpftime_trap_uprobe_attach_tests)

    add_library(bpftime_trap_late_module SHARED test/late_injection_module.cpp)
    target_link_libraries(bpftime_trap_late_module PRIVATE bpftime_trap_uprobe_attach_impl)
    target_compile_features(bpftime_trap_late_module PRIVATE cxx_std_20)
    foreach(symbol malloc calloc realloc free _Znwm pthread_mutex_lock __tls_get_addr)
        target_link_options(bpftime_trap_late_module PRIVATE -Wl,--wrap=${symbol})
    endforeach()
    add_executable(bpftime_trap_late_host test/late_injection_host.cpp)
    target_link_libraries(bpftime_trap_late_host PRIVATE dl pthread)
    target_compile_features(bpftime_trap_late_host PRIVATE cxx_std_20)
    add_dependencies(bpftime_trap_late_host bpftime_trap_late_module)
    add_dependencies(bpftime_trap_uprobe_attach_tests bpftime_trap_late_host)
    add_test(NAME bpftime_trap_late_injection
        COMMAND bpftime_trap_late_host $<TARGET_FILE:bpftime_trap_late_module>)
    set_tests_properties(bpftime_trap_late_injection PROPERTIES TIMEOUT 30)
endif()
