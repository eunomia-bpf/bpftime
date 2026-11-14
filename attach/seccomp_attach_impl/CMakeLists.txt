# Seccomp user-notification attach implementation.
# Linked into bpftime-agent like every other attach module; not a standalone
# shared object. It must not link bpftime-syscall-server and reaches the fake
# kernel only through the get_call_hook() dispatcher seam.
add_library(bpftime_seccomp_attach_impl STATIC
	src/seccomp_raw_syscall.cpp
	src/seccomp_notifier.cpp
)
target_include_directories(bpftime_seccomp_attach_impl
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../text_segment_transformer
)
target_link_libraries(bpftime_seccomp_attach_impl PRIVATE spdlog::spdlog)
set_property(TARGET bpftime_seccomp_attach_impl PROPERTY CXX_STANDARD 20)

if(BPFTIME_ENABLE_UNIT_TESTING AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  add_executable(bpftime_seccomp_failure_tests test/test_seccomp_failure.cpp)
  target_include_directories(bpftime_seccomp_failure_tests PRIVATE
    include ../text_segment_transformer)
  target_link_libraries(bpftime_seccomp_failure_tests PRIVATE
    Catch2::Catch2WithMain spdlog::spdlog)
  target_link_options(bpftime_seccomp_failure_tests PRIVATE
    -Wl,--wrap=ioctl -Wl,--wrap=prctl)
  add_test(NAME bpftime_seccomp_failure_tests COMMAND bpftime_seccomp_failure_tests)
endif()
