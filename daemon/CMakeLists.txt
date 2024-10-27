# Create a target that builds the ebpf program
add_ebpf_program_target(bpftime_daemon_ebpf_target ${CMAKE_CURRENT_SOURCE_DIR}/kernel/bpf_tracer.bpf.c ${CMAKE_CURRENT_BINARY_DIR}/bpf_tracer.bpf.o)

# Create a target that generated the bpf skeleton
add_bpf_skel_generating_target(bpftime_daemon_ebpf_skel ${CMAKE_CURRENT_BINARY_DIR}/bpf_tracer.bpf.o ${CMAKE_CURRENT_BINARY_DIR}/bpf_tracer.skel.h)

add_dependencies(bpftime_daemon_ebpf_skel bpftime_daemon_ebpf_target)

add_executable(embedfile ${CMAKE_CURRENT_SOURCE_DIR}/assets/embedfile.c)

set(PLACEHOLDER_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/assets/placeholder)
set(PLACEHOLDER_OUTPUT placeholder.c)

add_custom_command(
    OUTPUT placeholder.c
    COMMAND embedfile placeholder ${PLACEHOLDER_INPUT} ${PLACEHOLDER_OUTPUT}
    DEPENDS ${PLACEHOLDER_INPUT})

add_library(libbpftime_daemon STATIC
    user/bpf_tracer.cpp
    user/handle_bpf_event.cpp
    user/bpftime_driver.cpp
    ${PLACEHOLDER_OUTPUT}
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
    ${LIBBPF_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../vm/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/libbpf/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/libbpf/include/uapi
)
target_link_libraries(libbpftime_daemon PRIVATE
    ${LIBBPF_LIBRARIES}
    elf
    z
    spdlog::spdlog
    runtime
    PUBLIC
    rt
)

set_property(TARGET libbpftime_daemon PROPERTY CXX_STANDARD 20)

add_dependencies(bpftime_daemon libbpftime_daemon)
target_link_libraries(bpftime_daemon PRIVATE libbpftime_daemon)

install(TARGETS bpftime_daemon CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)

if(BPFTIME_ENABLE_UNIT_TESTING)
    add_subdirectory(test)
endif()

if(${TEST_LCOV})
    target_compile_options(bpftime_daemon PUBLIC -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
    target_link_options(bpftime_daemon PUBLIC -fprofile-arcs)
endif()
