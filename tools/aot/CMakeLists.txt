add_executable(
    bpftime-aot-cli
    main.cpp
)

set_target_properties(bpftime-aot-cli PROPERTIES OUTPUT_NAME "bpftime-aot")

target_include_directories(bpftime-aot-cli PRIVATE 
    ${SPDLOG_INCLUDE} 
    ${argparse_INCLUDE} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vm/compat/include
    ../../runtime/include/
    ../../runtime/src/
    ${LIBBPF_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)
if(${BPFTIME_ENABLE_CUDA_ATTACH})
    include(../../cmake/cuda.cmake)
    find_cuda()
    target_link_directories(bpftime-aot-cli PRIVATE ${CUDA_LIBRARY_PATH})
    # Use --whole-archive to include all symbols from bpftime_llvm_vm
    # This ensures the constructor function register_llvm_vm_factory() is linked
    target_link_options(bpftime-aot-cli PRIVATE 
        "-Wl,--whole-archive" 
        "$<TARGET_FILE:bpftime_llvm_vm>" 
        "-Wl,--no-whole-archive")
    target_link_libraries(bpftime-aot-cli PRIVATE spdlog::spdlog argparse bpftime_vm_compat bpftime_llvm_vm runtime ${LIBBPF_LIBRARIES} elf z ${LLVM_LIBRARIES} ${llvm_libs} ${CUDA_LIBS} bpftime_nv_attach_impl)
else()
    # Use --whole-archive to include all symbols from bpftime_llvm_vm
    # This ensures the constructor function register_llvm_vm_factory() is linked
    target_link_options(bpftime-aot-cli PRIVATE 
        "-Wl,--whole-archive" 
        "$<TARGET_FILE:bpftime_llvm_vm>" 
        "-Wl,--no-whole-archive")
    target_link_libraries(bpftime-aot-cli PRIVATE spdlog::spdlog argparse bpftime_vm_compat bpftime_llvm_vm runtime ${LIBBPF_LIBRARIES} elf z ${LLVM_LIBRARIES} ${llvm_libs})
endif()
set_property(TARGET bpftime-aot-cli PROPERTY CXX_STANDARD 20)

target_compile_definitions(bpftime-aot-cli PRIVATE _GNU_SOURCE)

target_link_directories(bpftime-aot-cli PRIVATE ${LLVM_LIBRARY_DIRS})

add_dependencies(bpftime-aot-cli spdlog::spdlog argparse bpftime_vm_compat bpftime_llvm_vm libbpf)

install(TARGETS bpftime-aot-cli CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)
