# Create a target that builds the ebpf program
add_ebpf_program_target(bpftime_daemon_ebpf_target ${CMAKE_CURRENT_SOURCE_DIR}/kernel/bpf_tracer.bpf.c ${CMAKE_CURRENT_BINARY_DIR}/bpf_tracer.bpf.o)

# Create a target that generated the bpf skeleton
add_bpf_skel_generating_target(bpftime_daemon_ebpf_skel ${CMAKE_CURRENT_BINARY_DIR}/bpf_tracer.bpf.o ${CMAKE_CURRENT_BINARY_DIR}/bpf_tracer.skel.h)

add_dependencies(bpftime_daemon_ebpf_skel bpftime_daemon_ebpf_target)

add_library(libbpftime_daemon STATIC     
    user/bpf_tracer.cpp 
    user/handle_bpf_event.cpp
    user/bpftime_driver.cpp
)

add_executable(bpftime_daemon
    user/main.cpp
)

add_dependencies(libbpftime_daemon 
        bpftime_daemon_ebpf_skel 
        libbpf 
        spdlog::spdlog
        runtime
    )

target_include_directories(libbpftime_daemon PRIVATE 
    ${CMAKE_CURRENT_BINARY_DIR} 
    ${LIBBPF_INCLUDE_DIRS}/uapi
    ${LIBBPF_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../vm/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include
)
target_link_libraries(libbpftime_daemon PRIVATE 
        ${LIBBPF_LIBRARIES}
        elf
        z
        spdlog::spdlog
        runtime
    )
set_property(TARGET libbpftime_daemon PROPERTY CXX_STANDARD 20)

add_dependencies(bpftime_daemon libbpftime_daemon)
target_link_libraries(bpftime_daemon PRIVATE libbpftime_daemon)

install(TARGETS bpftime_daemon CONFIGURATIONS Release Debug DESTINATION ~/.bpftime)

add_subdirectory(test)
