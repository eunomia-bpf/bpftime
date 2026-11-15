add_library(bpftime_text_segment_transformer SHARED
    agent-transformer.cpp
    text_segment_transformer.cpp
)
add_dependencies(bpftime_text_segment_transformer spdlog::spdlog FridaGum)
set_target_properties(bpftime_text_segment_transformer PROPERTIES CXX_STANDARD 20 OUTPUT_NAME "bpftime-agent-transformer")

# This library is the frida/capstone text-segment backend for the in-process
# uprobe path only. Rootless seccomp capture moved to runtime/syscall-server,
# so it must not link any seccomp/attach capture component anymore.
target_link_libraries(bpftime_text_segment_transformer
    spdlog::spdlog
    ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a
)

target_include_directories(bpftime_text_segment_transformer
    PRIVATE
    ${FRIDA_GUM_INSTALL_DIR}
    ${SPDLOG_INCLUDE}
)

# The trampoline defines its entry points from inline assembly only, so a
# link-time-optimized build drops them unless they carry `used` (issue #83).
# This target takes each of their addresses, so it fails to link when one of
# them disappears.
add_executable(bpftime_lto_link_check
    lto_link_check.cpp
    text_segment_transformer.cpp
)
add_dependencies(bpftime_lto_link_check spdlog::spdlog FridaGum)
set_target_properties(bpftime_lto_link_check PROPERTIES CXX_STANDARD 20)

target_link_libraries(bpftime_lto_link_check
    spdlog::spdlog
    ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a
)

target_include_directories(bpftime_lto_link_check
    PRIVATE
    ${FRIDA_GUM_INSTALL_DIR}
    ${SPDLOG_INCLUDE}
)

add_test(NAME lto_link_check COMMAND bpftime_lto_link_check)

if (${TEST_LCOV}) 
    target_compile_options(bpftime_text_segment_transformer PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
    target_link_options(bpftime_text_segment_transformer PRIVATE -fprofile-arcs)
endif()
