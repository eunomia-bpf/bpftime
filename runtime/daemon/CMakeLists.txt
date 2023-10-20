# Create a target that builds the ebpf program
add_ebpf_program_target(bpftime_daemon_ebpf_target ${CMAKE_CURRENT_SOURCE_DIR}/bpf-mocker.bpf.c ${CMAKE_CURRENT_BINARY_DIR}/bpf-mocker.bpf.o)

# Create a target that generated the bpf skeleton
add_bpf_skel_generating_target(bpftime_daemon_ebpf_skel ${CMAKE_CURRENT_BINARY_DIR}/bpf-mocker.bpf.o ${CMAKE_CURRENT_BINARY_DIR}/bpf-mocker.skel.h)

add_dependencies(bpftime_daemon_ebpf_skel bpftime_daemon_ebpf_target)

add_executable(bpftime_daemon bpf-mocker.cpp handle_bpf_event.cpp)
add_dependencies(bpftime_daemon bpftime_daemon_ebpf_skel libbpf)

target_include_directories(bpftime_daemon PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${LIBBPF_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bpftime_daemon PRIVATE ${LIBBPF_LIBRARIES} elf z)
set_property(TARGET bpftime_daemon PROPERTY CXX_STANDARD 20)
