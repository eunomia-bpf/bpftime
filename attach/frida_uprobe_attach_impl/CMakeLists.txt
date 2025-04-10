# include(../../cmake/frida.cmake)
add_library(
    bpftime_frida_uprobe_attach_impl STATIC
    src/frida_uprobe_attach_impl.cpp
    src/frida_internal_attach_entry.cpp
    src/frida_attach_entry.cpp
    src/frida_attach_utils.cpp
    src/frida_attach_private_data.cpp
    src/frida_register_conversion.cpp
)
add_dependencies(bpftime_frida_uprobe_attach_impl bpftime_base_attach_impl FridaGum spdlog::spdlog)

set(FRIDA_UPROBE_ATTACH_IMPL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE STRING "Include path of frida uprobe impl")

target_include_directories(bpftime_frida_uprobe_attach_impl PRIVATE ${SPDLOG_INCLUDE} ${FRIDA_UPROBE_ATTACH_IMPL_INCLUDE} PUBLIC ${FRIDA_GUM_INSTALL_DIR} ${BASE_ATTACH_IMPL_INCLUDE})

target_link_libraries(bpftime_frida_uprobe_attach_impl PRIVATE ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a PUBLIC bpftime_base_attach_impl spdlog::spdlog dl)

set(TEST_SOURCES
    test/test_uprobe_uretprobe.cpp
    test/test_function_address_resolve.cpp
    test/test_filter_attach_with_override.cpp
    test/test_replace_attach_with_override.cpp
    test/test_attach_with_unified_interface.cpp
    test/test_attach_private_data_parsing.cpp
    test/test_base_attach_impl.cpp
)
option(TEST_LCOV "option for lcov" OFF)
add_executable(bpftime_frida_uprobe_attach_tests ${TEST_SOURCES})

if (${TEST_LCOV}) 
    target_compile_options(bpftime_frida_uprobe_attach_tests PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
endif()

if(${ENABLE_EBPF_VERIFIER} AND NOT TARGET Catch2)
    message(STATUS "Adding Catch2 by FetchContent for frida_uprobe_attach_impl")
    Include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)

    # if not enable verifier, we will use the catch2 from submodule
endif()

add_dependencies(bpftime_frida_uprobe_attach_tests Catch2 bpftime_frida_uprobe_attach_impl spdlog::spdlog)
if (${TEST_LCOV}) 
    target_link_options(bpftime_frida_uprobe_attach_tests PRIVATE -lgcov)
    target_link_libraries(bpftime_frida_uprobe_attach_tests PRIVATE Catch2::Catch2WithMain bpftime_frida_uprobe_attach_impl spdlog::spdlog gcov)
else ()
    target_link_libraries(bpftime_frida_uprobe_attach_tests PRIVATE Catch2::Catch2WithMain bpftime_frida_uprobe_attach_impl spdlog::spdlog)
endif()
target_include_directories(bpftime_frida_uprobe_attach_tests PRIVATE ${FRIDA_UPROBE_ATTACH_IMPL_INCLUDE} ${Catch2_INCLUDE} ${SPDLOG_INCLUDE})

add_test(NAME bpftime_frida_uprobe_attach_tests COMMAND bpftime_frida_uprobe_attach_tests)

set_property(TARGET bpftime_frida_uprobe_attach_tests bpftime_frida_uprobe_attach_impl PROPERTY CXX_STANDARD 20)
