# SPDX-License-Identifier: Apache-2.0
#
# bpftime in-process, no-frida embed library. The alloc shim dlopen()s
# this shared object and drives the attached eBPF programs directly, so the
# workload never creates a frida interception point. It reuses the core
# runtime (VM, maps, shm, attach context) and only adds a thin C ABI.
option(BPFTIME_ENABLE_INPROC_EMBED "Build the bpftime in-process embed library" ON)

if(BPFTIME_ENABLE_INPROC_EMBED)
  add_library(bpftime_inproc_embed SHARED
    inproc_embed.cpp
  )
  set_target_properties(bpftime_inproc_embed PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME bpftime-inproc-embed
  )
  # `runtime` publicly propagates the VM, spdlog, base/simple attach headers and
  # the generated config include. The embed ABI only needs dlopen/dladdr on top.
  target_link_libraries(bpftime_inproc_embed
    PRIVATE
    runtime
    ${CMAKE_DL_LIBS}
  )
  target_include_directories(bpftime_inproc_embed
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  install(TARGETS bpftime_inproc_embed
    LIBRARY DESTINATION lib
  )
endif()

