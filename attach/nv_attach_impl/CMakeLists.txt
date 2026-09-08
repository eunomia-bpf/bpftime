include(../../cmake/cuda.cmake)

# PTX transform passes now live under pass/
add_subdirectory(pass/ptxpass_core)
add_subdirectory(pass/ptxpass_kprobe_entry)
add_subdirectory(pass/ptxpass_kretprobe)
add_subdirectory(pass/ptxpass_kprobe_memcapture)

list(APPEND PTX_PASS_EXECUTABLE $<TARGET_FILE:ptxpass_kprobe_entry>)
list(APPEND PTX_PASS_EXECUTABLE $<TARGET_FILE:ptxpass_kretprobe>)
list(APPEND PTX_PASS_EXECUTABLE $<TARGET_FILE:ptxpass_kprobe_memcapture>)

message("PTX_PASS_EXECUTABLE=${PTX_PASS_EXECUTABLE}")


# Independent PTX compiler
add_subdirectory(ptx_compiler)

list(APPEND PTX_COMPILER_SHARED_LIB $<TARGET_FILE:nv_attach_impl_ptx_compiler>)

message("PTX_COMPILER_SHARED_LIB=${PTX_COMPILER_SHARED_LIB}")

add_library(
    bpftime_nv_attach_impl STATIC
    nv_attach_impl.cpp
    nv_attach_private_data.cpp
    nv_attach_impl_frida_setup.cpp
    nv_attach_impl_patcher.cpp
    nv_attach_utils.cpp
    nv_attach_fatbin_record.cpp
    nv_elf_introspect.cpp
)

add_dependencies(
    bpftime_nv_attach_impl
    bpftime_base_attach_impl
    spdlog::spdlog
    FridaGum
    llvmbpf_vm
    bpftime_vm_compat
    ptxpass_core
    ptxpass_kprobe_entry
    ptxpass_kretprobe
    ptxpass_kprobe_memcapture
    nv_attach_impl_ptx_compiler
)

# Optional: regenerate the embedded trampoline PTX header
# (attach/nv_attach_impl/trampoline_ptx.h) from trampoline/default_trampoline.cu
# with the existing trampoline/Makefile generator BEFORE bpftime_nv_attach_impl
# is compiled. The embedded header carries the device-side map ABI layout; a
# stale embedded header once silently mismatched the 48-byte host MapBasicInfo.
# Default OFF: ordinary supported builds keep using the checked-in header and
# do not need the trampoline generator toolchain.
option(BPFTIME_REGENERATE_TRAMPOLINE_PTX
    "Regenerate trampoline_ptx.h via trampoline/Makefile before building bpftime_nv_attach_impl (requires a CUDA-capable clang)"
    OFF)
if(BPFTIME_REGENERATE_TRAMPOLINE_PTX)
    find_program(BPFTIME_TRAMPOLINE_MAKE NAMES make gmake)
    if(NOT BPFTIME_TRAMPOLINE_MAKE)
        message(FATAL_ERROR
            "BPFTIME_REGENERATE_TRAMPOLINE_PTX=ON needs GNU make (make/gmake); CMAKE_MAKE_PROGRAM is generator-specific (ninja under Ninja) and cannot run a Makefile")
    endif()
    set(BPFTIME_TRAMPOLINE_CLANG_FLAGS "" CACHE STRING
        "Optional CLANG_FLAGS override passed to trampoline/Makefile; empty keeps the Makefile's portable defaults")
    set(BPFTIME_TRAMPOLINE_CLANG_FLAGS_ARG)
    if(BPFTIME_TRAMPOLINE_CLANG_FLAGS)
        set(BPFTIME_TRAMPOLINE_CLANG_FLAGS_ARG
            "CLANG_FLAGS=${BPFTIME_TRAMPOLINE_CLANG_FLAGS}")
    endif()
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/trampoline_ptx.h
        # Bounded forced regeneration: CMake runs this command only when the
        # OUTPUT header is stale vs DEPENDS (no unconditional rebuild), and
        # GNU make -B forces the nested .cu -> .s -> header chain so a
        # Makefile-only change cannot be skipped as up-to-date inside.
        COMMAND ${BPFTIME_TRAMPOLINE_MAKE} -C ${CMAKE_CURRENT_SOURCE_DIR}/trampoline
            -B ../trampoline_ptx.h
            ${BPFTIME_TRAMPOLINE_CLANG_FLAGS_ARG}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/trampoline/default_trampoline.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/trampoline/Makefile
        COMMENT "Regenerating embedded trampoline PTX (trampoline/Makefile)"
        VERBATIM)
    add_custom_target(regenerate_trampoline_ptx
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/trampoline_ptx.h
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/trampoline/default_trampoline.cu)
    add_dependencies(bpftime_nv_attach_impl regenerate_trampoline_ptx)
endif()
set(NV_ATTACH_IMPL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} )
find_cuda()

target_include_directories(bpftime_nv_attach_impl PRIVATE ${SPDLOG_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR}/third_party PUBLIC ${CMAKE_SOURCE_DIR}/runtime/ ${CMAKE_SOURCE_DIR}/runtime/include ${BASE_ATTACH_IMPL_INCLUDE} ${NV_ATTACH_IMPL_INCLUDE} ${CUDA_INCLUDE_PATH} ${FRIDA_GUM_INSTALL_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime ${CMAKE_CURRENT_SOURCE_DIR}/../../vm/llvm-jit/include ${CMAKE_CURRENT_SOURCE_DIR}/pass/ptxpass_core/include)

target_link_directories(bpftime_nv_attach_impl PUBLIC ${CUDA_LIBRARY_PATH} )

target_link_libraries(bpftime_nv_attach_impl PUBLIC bpftime_base_attach_impl spdlog::spdlog ${CUDA_LIBS} PRIVATE  ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a PUBLIC llvmbpf_vm bpftime_vm_compat ptxpass_core)

set_property(TARGET bpftime_nv_attach_impl PROPERTY CXX_STANDARD 20)

if(${ENABLE_EBPF_VERIFIER})
  target_link_libraries(bpftime_nv_attach_impl PRIVATE bpftime-verifier)
  target_compile_definitions(bpftime_nv_attach_impl PRIVATE ENABLE_BPFTIME_VERIFIER)
endif()

# nv_attach_impl includes internal llvmbpf headers that include LLVM headers
# (e.g. vm/llvm-jit/src/llvm_jit_context.hpp). Make sure LLVM include dirs are
# available to this target as well.
if(DEFINED LLVM_INCLUDE_DIRS)
  target_include_directories(bpftime_nv_attach_impl PRIVATE ${LLVM_INCLUDE_DIRS})
endif()
get_target_property(LLVMBPF_VM_INCLUDES llvmbpf_vm INCLUDE_DIRECTORIES)
if(LLVMBPF_VM_INCLUDES)
  target_include_directories(bpftime_nv_attach_impl PRIVATE ${LLVMBPF_VM_INCLUDES})
endif()

# Create a build-time header file with the PTX pass paths
# This avoids shell escaping issues with colons and quotes
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ptx_pass_config.h"
     CONTENT "#pragma once\nstatic constexpr const char *DEFAULT_PTX_PASS_EXECUTABLE = \"$<JOIN:${PTX_PASS_EXECUTABLE},:>\";static constexpr const char* DEFAULT_PTX_COMPILER_SHARED_LIB=\"$<JOIN:${PTX_COMPILER_SHARED_LIB},:>\";\n")


target_link_options(bpftime_nv_attach_impl PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_nv_attach_impl>" "-Wl,--no-whole-archive")

add_subdirectory(test)


find_package(OpenSSL REQUIRED)

target_link_libraries(bpftime_nv_attach_impl PRIVATE OpenSSL::SSL OpenSSL::Crypto)
target_include_directories(bpftime_nv_attach_impl PRIVATE ${OPENSSL_INCLUDE_DIR})
