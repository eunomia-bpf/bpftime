add_executable(
    bpftime-vm-cli
    main.cpp
)

set_target_properties(bpftime-vm-cli PROPERTIES OUTPUT_NAME "bpftime-vm")

target_include_directories(bpftime-vm-cli 
    PRIVATE 
    ${SPDLOG_INCLUDE} 
    ${argparse_INCLUDE} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../llvm-jit/src 
    ${CMAKE_CURRENT_SOURCE_DIR}/../llvm-jit/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include 
    ${LIBBPF_INCLUDE_DIRS}
)
target_link_libraries(bpftime-vm-cli PRIVATE spdlog::spdlog argparse bpftime_llvm_jit_vm ${LIBBPF_LIBRARIES} elf z)
set_property(TARGET bpftime-vm-cli PROPERTY CXX_STANDARD 20)

target_compile_definitions(bpftime-vm-cli PRIVATE _GNU_SOURCE)

add_dependencies(bpftime-vm-cli spdlog::spdlog argparse bpftime_llvm_jit_vm libbpf)

install(TARGETS bpftime-vm-cli CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)
