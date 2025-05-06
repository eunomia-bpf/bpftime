include(../../cmake/cuda.cmake)

add_library(
    bpftime_nv_attach_impl STATIC
    nv_attach_impl.cpp
    ptx_load_mocker.cpp
    nv_attach_private_data.cpp
    cuda_injector.cpp
)

add_dependencies(bpftime_nv_attach_impl bpftime_base_attach_impl spdlog::spdlog FridaGum)
set(NV_ATTACH_IMPL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR})
find_cuda()

target_include_directories(bpftime_nv_attach_impl PRIVATE ${SPDLOG_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR} PUBLIC ${CMAKE_SOURCE_DIR}/runtime/ ${BASE_ATTACH_IMPL_INCLUDE} ${NV_ATTACH_IMPL_INCLUDE} ${CUDA_INCLUDE_PATH} ${FRIDA_GUM_INSTALL_DIR})

target_link_directories(bpftime_nv_attach_impl PUBLIC ${CUDA_LIBRARY_PATH} pos)

target_link_libraries(bpftime_nv_attach_impl PUBLIC bpftime_base_attach_impl spdlog::spdlog ${CUDA_LIBS} PRIVATE ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a)

set_property(TARGET bpftime_nv_attach_impl PROPERTY CXX_STANDARD 20)

set(TEST_SOURCES
    test_cuda_injector.cpp
)
option(TEST_LCOV "option for lcov" OFF)
add_executable(bpftime_nv_attach_tests ${TEST_SOURCES})

if(${TEST_LCOV})
    target_compile_options(bpftime_nv_attach_tests PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
endif()

if(${ENABLE_EBPF_VERIFIER} AND NOT TARGET Catch2)
    message(STATUS "Adding Catch2 by FetchContent at nv_attach_impl")
    Include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)

    # if not enable verifier, we will use the catch2 from submodule
endif()

add_dependencies(bpftime_nv_attach_tests Catch2 bpftime_nv_attach_impl spdlog::spdlog)
if(${TEST_LCOV})
    target_link_options(bpftime_nv_attach_tests PRIVATE -lgcov)
    target_link_libraries(bpftime_nv_attach_tests PRIVATE Catch2::Catch2WithMain bpftime_nv_attach_impl spdlog::spdlog gcov cuda pos)
else()
    target_link_libraries(bpftime_nv_attach_tests PRIVATE Catch2::Catch2WithMain bpftime_nv_attach_impl spdlog::spdlog cuda pos)
endif()
target_include_directories(bpftime_nv_attach_tests PRIVATE ${CMAKE_SOURCE_DIR}/runtime/ ${NV_ATTACH_IMPL_INCLUDE} ${Catch2_INCLUDE} ${SPDLOG_INCLUDE})

add_test(NAME bpftime_nv_attach_tests COMMAND bpftime_nv_attach_tests)

set_property(TARGET bpftime_nv_attach_tests PROPERTY CXX_STANDARD 20)

target_link_options(bpftime_nv_attach_impl PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_nv_attach_impl>" "-Wl,--no-whole-archive")
