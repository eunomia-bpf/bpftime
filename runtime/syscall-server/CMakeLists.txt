project(
    "bpftime-syscall-polyfill"
    VERSION 0.9.0
    LANGUAGES C CXX
)

# Rootless seccomp USER_NOTIF capture is built into this existing shared object.
# The notifier and the rootless __libc_start_main entry are frida/capstone-free
# and call only the local dispatch_syscall; the attach library is linked solely
# for the #815 tracefs id synthesis path, not for seccomp capture.
add_library(bpftime-syscall-server SHARED
    syscall_context.cpp
    syscall_server_main.cpp
    syscall_server_utils.cpp
    seccomp_notifier.cpp
    seccomp_raw_syscall.cpp
    rootless_agent_entry.cpp
)

# Force link VM compat libraries so constructor-based factory registration is retained
# Ensure VM compat libraries are linked (once) via the runtime library instead of duplicating at server
if(${BPFTIME_UBPF_JIT})
  add_dependencies(bpftime-syscall-server bpftime_ubpf_vm)
  target_link_options(bpftime-syscall-server PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_ubpf_vm>" "-Wl,--no-whole-archive")
  target_link_libraries(bpftime-syscall-server PUBLIC bpftime_ubpf_vm)
endif()

if(${BPFTIME_LLVM_JIT})
  add_dependencies(bpftime-syscall-server bpftime_llvm_vm llvmbpf_vm)
  target_link_libraries(bpftime-syscall-server PUBLIC bpftime_llvm_vm)
  target_sources(bpftime-syscall-server PRIVATE $<TARGET_OBJECTS:bpftime_llvm_vm_obj>)
endif()

target_link_libraries(bpftime-syscall-server PUBLIC
    runtime
    -lpthread
    -lm
    -ldl
    spdlog::spdlog)

# Tracepoint id synthesis (#815) is owned by the attach implementation; the
# syscall server links it for the mocked tracefs id path, while the in-object
# seccomp USER_NOTIF notifier (seccomp_notifier.cpp) stays frida/capstone-free.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(bpftime-syscall-server PRIVATE bpftime_syscall_trace_attach_impl)
endif()

add_dependencies(bpftime-syscall-server spdlog::spdlog)

target_include_directories(bpftime-syscall-server
    PUBLIC
    "../include"
    ${CMAKE_CURRENT_SOURCE_DIR}
    "../../vm"
    "../../third_party/libbpf/include/uapi"
    ${SPDLOG_INCLUDE}
)

if(${BPFTIME_BUILD_WITH_LIBBPF})
target_include_directories(bpftime-syscall-server PUBLIC
    "../../third_party/libbpf/include"
)
endif()

# Failure-path coverage for the in-object seccomp USER_NOTIF worker. Ported from
# the deleted attach/seccomp_attach_impl test; it compiles seccomp_notifier.cpp
# standalone with --wrap seams and mocks the object-local dispatch_syscall, so it
# links neither the full syscall-server shared object nor any attach/ library.
if(BPFTIME_ENABLE_UNIT_TESTING AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  add_executable(bpftime_seccomp_failure_tests test/test_seccomp_failure.cpp)
  target_include_directories(bpftime_seccomp_failure_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    "../include"
    ${SPDLOG_INCLUDE}
  )
  target_link_libraries(bpftime_seccomp_failure_tests PRIVATE
    Catch2::Catch2WithMain spdlog::spdlog)
  target_link_options(bpftime_seccomp_failure_tests PRIVATE
    -Wl,--wrap=ioctl -Wl,--wrap=prctl -Wl,--wrap=syscall)
  add_test(NAME bpftime_seccomp_failure_tests COMMAND bpftime_seccomp_failure_tests)
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
