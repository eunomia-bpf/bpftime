add_subdirectory(base_attach_impl)
add_subdirectory(frida_uprobe_attach_impl)
add_subdirectory(simple_attach_impl)
if(UNIX AND NOT APPLE)
  add_subdirectory(syscall_trace_attach_impl)
  add_subdirectory(text_segment_transformer)

  if(${BPFTIME_ENABLE_CUDA_ATTACH})
    add_subdirectory(nv_attach_impl)
  endif()
  
endif()
