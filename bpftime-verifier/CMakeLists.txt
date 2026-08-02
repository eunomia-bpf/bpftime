add_library(bpftime-verifier STATIC
    src/bpftime-verifier.cpp
    src/platform-impl.cpp
    src/gpu/gpu_platform.cpp
    src/gpu/gpu_verifier.cpp
    src/gpu/simt_safety_check.cpp
    src/gpu/uniformity_analysis.cpp
)
add_subdirectory(ebpf-verifier)

set_property(TARGET bpftime-verifier PROPERTY CXX_STANDARD 20)

target_include_directories(bpftime-verifier
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ebpf-verifier/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LIBBPF_INCLUDE_DIRS}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu
)

target_link_libraries(bpftime-verifier PUBLIC ebpfverifier ${LIBBPF_LIBRARIES} z elf)

add_dependencies(bpftime-verifier ebpfverifier libbpf)

set(BPFTIME_VERIFIER_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(TEST_SOURCES
    test/simple.cpp
    test/map.cpp
    test/non_kernel_helper.cpp
)

if(NOT TARGET Catch2)
    message(STATUS "Adding Catch2 by FetchContent at bpftime-verifier")
    Include(FetchContent)

    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.0.1
    )

    FetchContent_MakeAvailable(Catch2)
endif()

option(TEST_LCOV "option for lcov" OFF)
add_executable(bpftime_verifier_tests ${TEST_SOURCES})
add_executable(bpftime_gpu_verifier_tests test/gpu_verifier_test.cpp)

if (${TEST_LCOV}) 
    target_compile_options(bpftime_verifier_tests PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
    target_compile_options(bpftime_gpu_verifier_tests PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
endif()

add_dependencies(bpftime_verifier_tests bpftime-verifier)

if (${TEST_LCOV}) 
    target_link_options(bpftime_verifier_tests PRIVATE -lgcov)
    target_link_libraries(bpftime_verifier_tests PRIVATE bpftime-verifier Catch2::Catch2WithMain gcov)
    target_link_options(bpftime_gpu_verifier_tests PRIVATE -lgcov)
    target_link_libraries(bpftime_gpu_verifier_tests PRIVATE bpftime-verifier Catch2::Catch2WithMain gcov)
else ()
    target_link_libraries(bpftime_verifier_tests PRIVATE bpftime-verifier Catch2::Catch2WithMain)
    target_link_libraries(bpftime_gpu_verifier_tests PRIVATE bpftime-verifier Catch2::Catch2WithMain)
endif()
target_include_directories(bpftime_verifier_tests PRIVATE ${BPFTIME_VERIFIER_INCLUDE})
target_include_directories(bpftime_gpu_verifier_tests PRIVATE
    ${BPFTIME_VERIFIER_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu
)
add_test(NAME bpftime_verifier_tests COMMAND bpftime_verifier_tests)
add_test(NAME bpftime_gpu_verifier_tests COMMAND bpftime_gpu_verifier_tests)
