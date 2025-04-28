# add eBPF program target and controller executable target
add_executable(attach_impl_example_controller
    ./controller.cpp
)
add_ebpf_program_target(
    attach_impl_example_ebpf_program
    ${CMAKE_CURRENT_SOURCE_DIR}/request_filter.bpf.c
    ${CMAKE_CURRENT_BINARY_DIR}/request_filter.bpf.o
)
target_compile_definitions(attach_impl_example_controller PRIVATE EBPF_PROGRAM_PATH=${CMAKE_CURRENT_BINARY_DIR}/request_filter.bpf.o)

add_dependencies(attach_impl_example_controller runtime spdlog::spdlog bpftime-object attach_impl_example_ebpf_program)

target_link_libraries(attach_impl_example_controller PRIVATE runtime spdlog::spdlog bpftime-object)
target_include_directories(attach_impl_example_controller PRIVATE ${BPFTIME_RUNTIME_INCLUDE} ${SPDLOG_INCLUDE} ${BPFTIME_OBJECT_INCLUDE_DIRS})
