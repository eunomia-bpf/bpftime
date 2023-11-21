add_library(bpftime-verifier STATIC src/bpftime-verifier.cpp src/platform-impl.cpp)
add_subdirectory(ebpf-verifier)

set_property(TARGET bpftime-verifier PROPERTY CXX_STANDARD 20)

target_include_directories(bpftime-verifier
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ebpf-verifier/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LIBBPF_INCLUDE_DIRS}
)

target_link_libraries(bpftime-verifier PUBLIC ebpfverifier ${LIBBPF_LIBRARIES} z elf)

add_dependencies(bpftime-verifier ebpfverifier libbpf)

set(BPFTIME_VERIFIER_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(TEST_SOURCES
    test/simple.cpp
    test/map.cpp
    test/non_kernel_helper.cpp
)

add_executable(bpftime_verifier_tests ${TEST_SOURCES})
add_dependencies(bpftime_verifier_tests bpftime-verifier)
target_link_libraries(bpftime_verifier_tests PRIVATE bpftime-verifier Catch2::Catch2WithMain)
target_include_directories(bpftime_verifier_tests PRIVATE ${BPFTIME_VERIFIER_INCLUDE})
add_test(NAME bpftime_verifier_tests COMMAND bpftime_verifier_tests)
