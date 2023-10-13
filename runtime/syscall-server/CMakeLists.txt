project(
    "bpftime-syscall-polyfill"
    VERSION 0.1.0
    LANGUAGES C CXX
)

add_library(bpftime-syscall-server SHARED
    syscall_context.cpp
    syscall_server_main.cpp
    syscall_server_utils.cpp
)
target_link_libraries(bpftime-syscall-server PUBLIC
    runtime
    -lpthread
    -lm
    -ldl
    spdlog::spdlog)
add_dependencies(bpftime-syscall-server spdlog::spdlog)
target_include_directories(bpftime-syscall-server
    PUBLIC
    "../../core"
    "../../third_party/libbpf/include"
    "../../third_party/libbpf/include/uapi"
    ${SPDLOG_INCLUDE}
)
set_property(TARGET bpftime-syscall-server PROPERTY CXX_STANDARD 20)

if(${ENABLE_EBPF_VERIFIER})
    add_dependencies(bpftime-syscall-server bpftime-verifier)
    target_link_libraries(bpftime-syscall-server PRIVATE bpftime-verifier)
    target_include_directories(bpftime-syscall-server PRIVATE ${BPFTIME_VERIFIER_INCLUDE})
    target_compile_definitions(bpftime-syscall-server PRIVATE ENABLE_EBPF_VERIFIER ENABLE_BPFTIME_VERIFIER)
endif()
