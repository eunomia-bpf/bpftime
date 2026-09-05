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

# PTX-free host-application companion artifact: ptxas output is a SASS-only
# cubin (no PTX section), mirroring a shipping application that embeds
# precompiled SASS. Targets sm_120 to match the SassAotOptions::sm default.
set(SASS_AOT_PTXAS_BIN ${BPFTIME_CUDA_ROOT}/bin/ptxas)
if(NOT EXISTS ${SASS_AOT_PTXAS_BIN})
    find_program(SASS_AOT_PTXAS_BIN NAMES ptxas)
endif()
if(NOT SASS_AOT_PTXAS_BIN)
    message(FATAL_ERROR
        "sass_aot companion: ptxas not found (set BPFTIME_CUDA_ROOT)")
endif()
set(SASS_AOT_COMPANION_APP_PTX ${CMAKE_CURRENT_SOURCE_DIR}/companion_app_kernel.ptx)
set(SASS_AOT_COMPANION_APP_CUBIN ${CMAKE_CURRENT_BINARY_DIR}/companion_app_kernel.cubin)
add_custom_command(
    OUTPUT ${SASS_AOT_COMPANION_APP_CUBIN}
    COMMAND ${SASS_AOT_PTXAS_BIN} -arch=sm_120
        ${SASS_AOT_COMPANION_APP_PTX} -o ${SASS_AOT_COMPANION_APP_CUBIN}
    DEPENDS ${SASS_AOT_COMPANION_APP_PTX}
    COMMENT "Assembling PTX-free application companion cubin (ptxas -arch=sm_120)"
    VERBATIM
)
add_custom_target(sass_aot_companion_app_cubin DEPENDS ${SASS_AOT_COMPANION_APP_CUBIN})

add_executable(bpftime_sass_aot_tests test/test_sass_aot.cpp)
set_property(TARGET bpftime_sass_aot_tests PROPERTY CXX_STANDARD 20)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sass_aot_test_config.hpp"
    CONTENT "#pragma once\ninline constexpr const char *SASS_AOT_SPIKE_BPF_OBJECT = \"${SASS_AOT_BPF_OBJECT}\";\ninline constexpr const char *SASS_AOT_COMPANION_APP_CUBIN = \"${SASS_AOT_COMPANION_APP_CUBIN}\";\n")
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

# Deliberately not registered with CTest: explicit live-GPU acceptance
# executable for the companion/interposition boundary. It preserves the
# application's own PTX-free SASS module execution and invokes the
# BPF-derived entry through the documented host-side boundary.
add_executable(bpftime_sass_aot_companion_live companion_sass_aot_live.cpp)
set_property(TARGET bpftime_sass_aot_companion_live PROPERTY CXX_STANDARD 20)
target_include_directories(bpftime_sass_aot_companion_live PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(bpftime_sass_aot_companion_live PRIVATE bpftime_sass_aot)
add_dependencies(bpftime_sass_aot_companion_live
    sass_aot_spike_bpf_object sass_aot_companion_app_cubin)
