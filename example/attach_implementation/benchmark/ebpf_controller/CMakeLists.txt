# add eBPF program target and controller executable target
add_executable(nginx_benchmark_ebpf_controller
    ./controller.cpp
)
add_ebpf_program_target(
    nginx_benchmark_ebpf_program
    ${CMAKE_CURRENT_SOURCE_DIR}/request_filter.bpf.c
    ${CMAKE_CURRENT_BINARY_DIR}/request_filter.bpf.o
)
target_compile_definitions(nginx_benchmark_ebpf_controller PRIVATE EBPF_PROGRAM_PATH=${CMAKE_CURRENT_BINARY_DIR}/request_filter.bpf.o)

add_dependencies(nginx_benchmark_ebpf_controller runtime spdlog::spdlog bpftime-object nginx_benchmark_ebpf_program)

target_link_libraries(nginx_benchmark_ebpf_controller PRIVATE runtime spdlog::spdlog bpftime-object)
target_include_directories(nginx_benchmark_ebpf_controller PRIVATE ${BPFTIME_RUNTIME_INCLUDE} ${SPDLOG_INCLUDE} ${BPFTIME_OBJECT_INCLUDE_DIRS})
