# eBPF-to-SASS AOT build spike.
#
# Pipeline: real BPF ELF object -> strict GPU verifier ->
# ptxpass::compile_ebpf_to_ptx_from_words -> ptxas -arch=sm_120 ->
# cubin/SASS artifact, inspected with cuobjdump.
#
# Enabled only when BPFTIME_ENABLE_SASS_AOT_SPIKE=1 (default OFF), which
# requires BPFTIME_ENABLE_CUDA_ATTACH=1 and ENABLE_EBPF_VERIFIER=1.

add_library(bpftime_sass_aot STATIC sass_aot.cpp)
set_property(TARGET bpftime_sass_aot PROPERTY CXX_STANDARD 20)

target_include_directories(bpftime_sass_aot PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../pass/ptxpass_core/include
    ${CUDA_INCLUDE_PATH})

target_link_directories(bpftime_sass_aot PUBLIC ${CUDA_LIBRARY_PATH})
target_link_libraries(bpftime_sass_aot PUBLIC ptxpass_core bpftime-verifier cuda)

# Real BPF object consumed by the spike test
find_program(SASS_AOT_CLANG NAMES clang clang-18 clang-17 clang-15)
if(NOT SASS_AOT_CLANG)
    message(FATAL_ERROR
        "BPFTIME_ENABLE_SASS_AOT_SPIKE requires clang to build the spike BPF object")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(SASS_AOT_BPF_ARCH arm64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(SASS_AOT_BPF_ARCH x86)
else()
    message(FATAL_ERROR "sass_aot spike: unsupported host arch "
        "${CMAKE_SYSTEM_PROCESSOR}")
endif()
set(SASS_AOT_BPF_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/sass_aot_spike.bpf.o)

add_custom_command(
    OUTPUT ${SASS_AOT_BPF_OBJECT}
    COMMAND ${SASS_AOT_CLANG} -g -O2 -target bpf
        -D__TARGET_ARCH_${SASS_AOT_BPF_ARCH}
        -I${CMAKE_SOURCE_DIR}/third_party/vmlinux/${SASS_AOT_BPF_ARCH}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/sass_aot_spike.bpf.c
        -o ${SASS_AOT_BPF_OBJECT}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/sass_aot_spike.bpf.c
    COMMENT "Compiling spike BPF object (clang -target bpf)"
    VERBATIM
)
add_custom_target(sass_aot_spike_bpf_object DEPENDS ${SASS_AOT_BPF_OBJECT})

add_executable(bpftime_sass_aot_tests test/test_sass_aot.cpp)
set_property(TARGET bpftime_sass_aot_tests PROPERTY CXX_STANDARD 20)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sass_aot_test_config.hpp"
    CONTENT "#pragma once\ninline constexpr const char *SASS_AOT_SPIKE_BPF_OBJECT = \"${SASS_AOT_BPF_OBJECT}\";\n")
target_include_directories(bpftime_sass_aot_tests PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(bpftime_sass_aot_tests PRIVATE bpftime_sass_aot)
add_dependencies(bpftime_sass_aot_tests sass_aot_spike_bpf_object)

add_test(NAME bpftime_sass_aot_tests COMMAND bpftime_sass_aot_tests)

# Deliberately not registered with CTest: this is the explicit live-GPU
# acceptance executable for the standalone AOT path.
add_executable(bpftime_sass_aot_live live_sass_aot.cpp)
set_property(TARGET bpftime_sass_aot_live PROPERTY CXX_STANDARD 20)
target_include_directories(bpftime_sass_aot_live PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(bpftime_sass_aot_live PRIVATE bpftime_sass_aot)
add_dependencies(bpftime_sass_aot_live sass_aot_spike_bpf_object)
