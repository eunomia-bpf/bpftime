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

set_property(TARGET bpftime_trap_uprobe_attach_impl PROPERTY CXX_STANDARD 20)
set_property(TARGET bpftime_trap_uprobe_attach_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

if(BPFTIME_TRAP_ATTACH_BUILD_TESTS)
    set(TRAP_TEST_SOURCES
        test/test_trap_uprobe_uretprobe.cpp
        test/test_trap_override.cpp
        test/test_trap_unified_interface.cpp
        test/test_trap_private_data.cpp
        test/test_trap_function_resolve.cpp
        test/test_trap_multithread.cpp
        test/test_trap_signal_chaining.cpp
        test/test_trap_backtrace.cpp
        test/test_trap_recursion.cpp
        test/test_trap_insn_decode.cpp
        test/test_trap_signal_safety.cpp
        test/test_trap_concurrent_patch.cpp
    )
    add_executable(bpftime_trap_uprobe_attach_tests ${TRAP_TEST_SOURCES})
    add_dependencies(bpftime_trap_uprobe_attach_tests Catch2 bpftime_trap_uprobe_attach_impl spdlog::spdlog)
    target_link_libraries(bpftime_trap_uprobe_attach_tests PRIVATE Catch2::Catch2WithMain bpftime_trap_uprobe_attach_impl spdlog::spdlog pthread)
    target_include_directories(bpftime_trap_uprobe_attach_tests PRIVATE ${TRAP_UPROBE_ATTACH_IMPL_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/src ${Catch2_INCLUDE} ${SPDLOG_INCLUDE})
    set_property(TARGET bpftime_trap_uprobe_attach_tests PROPERTY CXX_STANDARD 20)
    add_test(NAME bpftime_trap_uprobe_attach_tests COMMAND bpftime_trap_uprobe_attach_tests)
endif()
