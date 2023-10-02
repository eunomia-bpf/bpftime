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
find_package(Catch2 REQUIRED)

foreach(test_file ${TEST_SOURCES})
    string(REGEX REPLACE "(.*/)([a-zA-Z0-9_ ]+)(\.cpp)" "\\2" test_file_replaced ${test_file})
    set(current_test_name bpftime_verifier_${test_file_replaced}_Tests)
    add_executable(${current_test_name} ${test_file})
    add_dependencies(${current_test_name} bpftime-verifier)
    target_link_libraries(
        ${current_test_name}
        PRIVATE
        bpftime-verifier
        Catch2::Catch2WithMain
    )
    target_include_directories(${current_test_name} PRIVATE ${BPFTIME_VERIFIER_INCLUDE})
    add_test(NAME ${current_test_name} COMMAND ${current_test_name} ${Catch2_INCLUDE})
    message(STATUS "Add test from bpftime-verifier: ${current_test_name}")
endforeach()
