add_subdirectory(base_attach_impl)
add_subdirectory(frida_uprobe_attach_impl)
# disable syscall trace when not build libbpf
if(BPFTIME_BUILD_WITH_LIBBPF)
  add_subdirectory(syscall_trace_attach_impl)
  add_subdirectory(text_segment_transformer)
endif()
