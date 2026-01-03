add_executable(simple-benchmark-with-embed-ebpf-calling test_embed.c)

add_dependencies(simple-benchmark-with-embed-ebpf-calling bpftime_vm libbpf)
target_compile_definitions(simple-benchmark-with-embed-ebpf-calling
    PRIVATE
)
target_link_libraries(simple-benchmark-with-embed-ebpf-calling bpftime_vm ${LIBBPF_LIBRARIES} elf z dl)
target_include_directories(simple-benchmark-with-embed-ebpf-calling
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}../vm/include
    ${LIBBPF_INCLUDE_DIRS})

if(BPFTIME_ENABLE_CUDA_ATTACH)
  add_executable(gpu_array_map_host_perf
    gpu/host/gpu_array_map_host_perf.cpp)
  add_dependencies(gpu_array_map_host_perf runtime)
  target_link_libraries(gpu_array_map_host_perf
    PRIVATE
    runtime
    spdlog::spdlog
  )
  set_target_properties(gpu_array_map_host_perf
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark/gpu/host
  )
  add_executable(gpu_per_thread_array_map_host_perf
    gpu/host/gpu_per_thread_array_map_host_perf.cpp)
  add_dependencies(gpu_per_thread_array_map_host_perf runtime)
  target_link_libraries(gpu_per_thread_array_map_host_perf
    PRIVATE
    runtime
    spdlog::spdlog
  )
  set_target_properties(gpu_per_thread_array_map_host_perf
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark/gpu/host
  )
endif()
