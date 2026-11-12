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
