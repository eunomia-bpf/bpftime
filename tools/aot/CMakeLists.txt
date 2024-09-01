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
    ${LIBBPF_INCLUDE_DIRS})
target_link_libraries(bpftime-aot-cli PRIVATE spdlog::spdlog argparse bpftime_vm_compat bpftime_llvm_vm runtime ${LIBBPF_LIBRARIES} elf z)
set_property(TARGET bpftime-aot-cli PROPERTY CXX_STANDARD 20)

target_compile_definitions(bpftime-aot-cli PRIVATE _GNU_SOURCE)

add_dependencies(bpftime-aot-cli spdlog::spdlog argparse bpftime_vm_compat bpftime_llvm_vm libbpf)

install(TARGETS bpftime-aot-cli CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)
