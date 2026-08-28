add_subdirectory(base_attach_impl)
add_subdirectory(frida_uprobe_attach_impl)
add_subdirectory(simple_attach_impl)

# Linux-only attach backends (syscall rewrite / CUDA). QNX phase-1 is uprobe only.
if(UNIX AND NOT APPLE AND NOT BPFTIME_TARGET_QNX)
  add_subdirectory(syscall_trace_attach_impl)
  add_subdirectory(text_segment_transformer)

  if(${BPFTIME_ENABLE_CUDA_ATTACH})
    add_subdirectory(nv_attach_impl)
  endif()
endif()
