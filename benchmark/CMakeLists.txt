add_executable(simple-benchmark-with-embed-ebpf-calling test_embed.c)

add_ebpf_program_target(uprobe_prog ${CMAKE_CURRENT_SOURCE_DIR}/uprobe/uprobe.bpf.c ${CMAKE_CURRENT_BINARY_DIR}/uprobe_prog.bpf.o)
add_ebpf_program_target(uretprobe_prog ${CMAKE_CURRENT_SOURCE_DIR}/uretprobe/uretprobe.bpf.c ${CMAKE_CURRENT_BINARY_DIR}/uretprobe_prog.bpf.o)

add_dependencies(simple-benchmark-with-embed-ebpf-calling vm-bpf uprobe_prog uretprobe_prog libbpf)
target_compile_definitions(simple-benchmark-with-embed-ebpf-calling
    PRIVATE
    UPROBE_PROG=${CMAKE_CURRENT_BINARY_DIR}/uprobe_prog.bpf.o
    URETPROBE_PROG=${CMAKE_CURRENT_BINARY_DIR}/uretprobe_prog.bpf.o
)
target_link_libraries(simple-benchmark-with-embed-ebpf-calling vm-bpf ${LIBBPF_LIBRARIES} elf z)
target_include_directories(simple-benchmark-with-embed-ebpf-calling
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}../vm/include
    ${LIBBPF_INCLUDE_DIRS})
