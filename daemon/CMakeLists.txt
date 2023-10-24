# Create a target that builds the ebpf program
add_ebpf_program_target(bpftime_daemon_ebpf_target ${CMAKE_CURRENT_SOURCE_DIR}/kernel/bpf_tracer.bpf.c ${CMAKE_CURRENT_BINARY_DIR}/bpf_tracer.bpf.o)

# Create a target that generated the bpf skeleton
add_bpf_skel_generating_target(bpftime_daemon_ebpf_skel ${CMAKE_CURRENT_BINARY_DIR}/bpf_tracer.bpf.o ${CMAKE_CURRENT_BINARY_DIR}/bpf_tracer.skel.h)

add_dependencies(bpftime_daemon_ebpf_skel bpftime_daemon_ebpf_target)

add_executable(bpftime_daemon
    user/main.cpp 
    user/bpf_tracer.cpp 
    user/handle_bpf_event.cpp)

add_dependencies(bpftime_daemon 
        bpftime_daemon_ebpf_skel 
        libbpf 
        spdlog::spdlog
        runtime
    )

target_include_directories(bpftime_daemon PRIVATE 
    ${CMAKE_CURRENT_BINARY_DIR} 
    ${LIBBPF_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../vm/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include
)
target_link_libraries(bpftime_daemon PRIVATE 
        ${LIBBPF_LIBRARIES}
        elf
        z
        spdlog::spdlog
        runtime
    )
set_property(TARGET bpftime_daemon PROPERTY CXX_STANDARD 20)

add_subdirectory(test)
