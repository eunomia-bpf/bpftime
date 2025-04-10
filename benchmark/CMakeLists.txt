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
