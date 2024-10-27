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

if(${BPFTIME_BUILD_WITH_LIBBPF})

target_include_directories(bpftime-syscall-server
    PUBLIC
    "../include"
    "../../vm"
    "../../third_party/libbpf/include"
    "../../third_party/libbpf/include/uapi"
    ${SPDLOG_INCLUDE}
)
else()
target_include_directories(bpftime-syscall-server
    PUBLIC
    "../../vm"
    ${SPDLOG_INCLUDE}
)
endif()

set_target_properties(bpftime-syscall-server PROPERTIES CXX_STANDARD 20 LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/syscall-server.version)

if(UNIX AND NOT APPLE)
target_link_options(bpftime-syscall-server PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/syscall-server.version)
endif()

if(${ENABLE_EBPF_VERIFIER})
    add_dependencies(bpftime-syscall-server bpftime-verifier)
    target_link_libraries(bpftime-syscall-server PRIVATE bpftime-verifier)
    target_include_directories(bpftime-syscall-server PRIVATE ${BPFTIME_VERIFIER_INCLUDE})
    target_compile_definitions(bpftime-syscall-server PRIVATE ENABLE_EBPF_VERIFIER ENABLE_BPFTIME_VERIFIER)
endif()

if (${TEST_LCOV}) 
    target_compile_options(bpftime-syscall-server PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
    target_link_options(bpftime-syscall-server PRIVATE -fprofile-arcs)
endif()
