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
    rootless_map_files.cpp
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

# ARCHITECTURE INVARIANT (enforced below): bpftime-syscall-server must never
# link or include any attach/ library (which may pull frida/capstone) and must
# never reference frida/capstone itself. Rootless tracepoint id synthesis is
# implemented locally in syscall_server_utils.cpp using only the runtime-level
# generated syscall name->number table (the syscall_id_table target).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_dependencies(bpftime-syscall-server syscall_id_table)
    target_include_directories(bpftime-syscall-server PRIVATE
        ${CMAKE_BINARY_DIR}/runtime)
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
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|aarch64|ARM64)$")
  add_executable(bpftime_seccomp_failure_tests
      test/test_seccomp_failure.cpp rootless_map_files.cpp)
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

# Rootless map_files resolution: pure fallback/validation logic plus a live
# /proc/<pid>/map_files range exercised without CAP_SYS_ADMIN. Links the real
# raw-syscall implementation instead of the --wrap seams used above.
if(BPFTIME_ENABLE_UNIT_TESTING AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(bpftime_rootless_map_files_tests
      test/test_rootless_map_files.cpp
      rootless_map_files.cpp
      seccomp_raw_syscall.cpp)
  target_include_directories(bpftime_rootless_map_files_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    "../include"
    ${SPDLOG_INCLUDE}
  )
  target_link_libraries(bpftime_rootless_map_files_tests PRIVATE
      Catch2::Catch2WithMain spdlog::spdlog)
  add_test(NAME bpftime_rootless_map_files_tests
           COMMAND bpftime_rootless_map_files_tests)
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

# ---------------------------------------------------------------------------
# Architecture guard: fail the build at configure time if the self-contained
# syscall server ever grows an attach/ or frida/capstone dependency again.
# This is a hard assertion, not a convention: it greps the actual sources and
# the link line of this target, so re-introducing the coupling cannot silently
# pass CI. The seccomp failure test (test/) is excluded: it intentionally
# #includes seccomp_notifier.cpp standalone with --wrap seams.
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  file(GLOB _syscall_server_sources
       CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
       "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
  # Plain substrings checked with string(FIND), not regex: regex tokens such
  # as [^\n] get mangled when expanded out of a CMake list in foreach. These
  # are real code dependencies (include directives, CMake target names and
  # C symbols); documentation comments that only mention the words are fine.
  # seccomp_tracer.hpp lives in this same syscall-server/ directory and is a
  # legitimate local include, so it is NOT forbidden. Only cross-directory
  # includes into attach/, attach CMake targets and frida/capstone C symbols
  # are violations.
  set(_forbidden_substrings
      "bpftime_syscall_trace_attach_impl"
      "bpftime_frida_uprobe_attach_impl"
      "bpftime_base_attach_impl"
      "frida_gum"
      "gum_interceptor"
      "capstone.h")
  foreach(_src ${_syscall_server_sources})
    file(READ "${_src}" _contents)
    # Cross-directory include into the attach tree (a local same-dir header
    # such as "seccomp_tracer.hpp" has no attach/ path segment and is fine).
    string(FIND "${_contents}" "#include \"../attach/" _attach_inc)
    string(FIND "${_contents}" "#include \"attach/" _attach_inc2)
    if(NOT _attach_inc EQUAL -1 OR NOT _attach_inc2 EQUAL -1)
      message(FATAL_ERROR
        "Architecture violation: ${_src} includes a header from attach/. "
        "bpftime-syscall-server must stay self-contained.")
    endif()
    foreach(_needle ${_forbidden_substrings})
      string(FIND "${_contents}" "${_needle}" _hit)
      if(NOT _hit EQUAL -1)
        message(FATAL_ERROR
          "Architecture violation: ${_src} contains forbidden dependency "
          "'${_needle}'. bpftime-syscall-server must stay self-contained with "
          "zero attach/frida/capstone dependency. See the invariant comment "
          "at the top of this CMakeLists.txt.")
      endif()
    endforeach()
  endforeach()
endif()
