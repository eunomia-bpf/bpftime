include(luthier.cmake)

if(BPFTIME_BUILD_LUTHIER_EXAMPLES)
    add_subdirectory(luthier-examples)
endif()

# The Luthier verifier can be built before enabling the ROCm attach backend.
if(NOT BPFTIME_ENABLE_ROCM_ATTACH)
    return()
endif()

include(../../cmake/rocm.cmake)

# Luthier is built by ExternalProject because its compiler plugins and HIP
# sources require ROCm's compiler. Its LLVM libraries are the same imported
# targets used by llvmbpf, selected at the root before either is configured.
# Keep this interface aligned with the pinned LuthierTooling target.
find_package(Threads REQUIRED)
find_package(hsa-runtime64 REQUIRED CONFIG
    PATHS "${BPFTIME_ROCM_ROOT}/lib/cmake/hsa-runtime64" NO_DEFAULT_PATH)
add_library(bpftime_luthier_tooling STATIC IMPORTED GLOBAL)
set_target_properties(bpftime_luthier_tooling PROPERTIES
    IMPORTED_LOCATION "${BPFTIME_LUTHIER_BUILD_DIR}/lib/libLuthierTooling.a")
add_dependencies(bpftime_luthier_tooling bpftime_luthier_backend)
target_link_libraries(bpftime_luthier_tooling INTERFACE
    ${CMAKE_DL_LIBS} Threads::Threads hsa-runtime64::hsa-runtime64
    LLVMAMDGPUAsmParser LLVMAMDGPUDesc LLVMAMDGPUDisassembler LLVMAMDGPUInfo
    LLVMAMDGPUTargetMCA LLVMAMDGPUUtils LLVMAMDGPUCodeGen
    LLVMSymbolize LLVMDebugInfoDWARF LLVMCodeGenTypes LLVMMC LLVMMCA
    LLVMMCDisassembler LLVMObject LLVMSupport LLVMTarget LLVMTargetParser
    LLVMTransformUtils LLVMBitReader LLVMAnalysis LLVMAsmPrinter LLVMCore
    LLVMPasses LLVMCodeGen LLVMLinker LLVMIRReader)

add_library(
    bpftime_rocm_attach_impl STATIC
    rocm_attach_impl.cpp
    rocm_attach_private_data.cpp
    rocm_profiler.cpp
)

add_dependencies(bpftime_rocm_attach_impl bpftime_base_attach_impl spdlog::spdlog llvmbpf_vm bpftime_vm_compat)
set(ROCM_ATTACH_IMPL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} )
find_rocm()

target_include_directories(bpftime_rocm_attach_impl PRIVATE ${SPDLOG_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/luthier/include" "${BPFTIME_LUTHIER_BUILD_DIR}/include"
    PUBLIC ${CMAKE_SOURCE_DIR}/runtime/ ${BASE_ATTACH_IMPL_INCLUDE} ${ROCM_ATTACH_IMPL_INCLUDE} ${ROCM_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime ${CMAKE_CURRENT_SOURCE_DIR}/../../vm/llvm-jit/include)

target_link_directories(bpftime_rocm_attach_impl PUBLIC ${ROCM_LIBRARY_PATH} )

target_link_libraries(bpftime_rocm_attach_impl PUBLIC bpftime_base_attach_impl spdlog::spdlog ${ROCM_LIBS} llvmbpf_vm bpftime_vm_compat)

# Retain Luthier's embedded device-code sections and implementation.
target_link_libraries(bpftime_rocm_attach_impl PRIVATE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,bpftime_luthier_tooling>")
target_include_directories(bpftime_rocm_attach_impl SYSTEM PRIVATE
    "${BPFTIME_ROCM_ROOT}/lib/llvm/include")
target_compile_definitions(bpftime_rocm_attach_impl PRIVATE AMD_INTERNAL_BUILD __HIP_PLATFORM_AMD__=1)

set_property(TARGET bpftime_rocm_attach_impl PROPERTY CXX_STANDARD 23)
# Only the Luthier-facing TU uses LLVM types; the attach implementation still
# needs RTTI for bpftime's attach_private_data interface.
if(NOT LLVM_ENABLE_RTTI)
    set_source_files_properties(rocm_profiler.cpp PROPERTIES COMPILE_OPTIONS -fno-rtti)
endif()

target_link_options(bpftime_rocm_attach_impl PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_rocm_attach_impl>" "-Wl,--no-whole-archive")
# ROCm also loads a shared LLVM through COMGR. Keep the embedded static
# implementation private in every consumer (including CLI/daemon executables),
# otherwise ELF symbol interposition can mix their global objects and crash
# during destruction. The agent's version script alone cannot protect tools.
target_link_options(bpftime_rocm_attach_impl PUBLIC -Wl,--exclude-libs,ALL)

if(BPFTIME_ENABLE_UNIT_TESTING)
    add_subdirectory(test)
endif()
