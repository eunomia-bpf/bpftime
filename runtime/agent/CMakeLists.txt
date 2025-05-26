add_library(bpftime-agent SHARED
  agent.cpp
)

if(NOT ${BPFTIME_LLVM_JIT})
  message(FATAL_ERROR "LLVM_JIT must be enabled to use bpftime-super")
endif()

if(${BPFTIME_BUILD_WITH_LIBBPF})
  add_dependencies(bpftime-agent FridaGum spdlog::spdlog bpftime_frida_uprobe_attach_impl bpftime_syscall_trace_attach_impl)
else()
  add_dependencies(bpftime-agent FridaGum spdlog::spdlog bpftime_frida_uprobe_attach_impl)
endif()

set_target_properties(bpftime-agent PROPERTIES CXX_STANDARD 20 LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/agent.version)

if(UNIX AND NOT APPLE)
  target_link_options(bpftime-agent PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/agent.version)
endif()

if(${TEST_LCOV})
  target_compile_options(bpftime-agent PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
  target_link_options(bpftime-agent PRIVATE -fprofile-arcs)
endif()

add_dependencies(bpftime-agent bpftime_llvm_vm llvmbpf_vm bpftime_vm)

if(${BPFTIME_BUILD_WITH_LIBBPF})
  target_include_directories(bpftime-agent
    PRIVATE
    ${FRIDA_GUM_INSTALL_DIR}
    ../include
    ../../third_party/
    ${SPDLOG_INCLUDE}
    ${FRIDA_UPROBE_ATTACH_IMPL_INCLUDE}
    ${SYSCALL_TRACE_ATTACH_IMPL_INCLUDE}
    ${NV_ATTACH_IMPL_INCLUDE}
  )
  target_link_libraries(bpftime-agent
    PUBLIC
    ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a
    runtime
    -lpthread
    -lm
    -ldl
    spdlog::spdlog
    bpftime_frida_uprobe_attach_impl
    bpftime_syscall_trace_attach_impl
    # bpftime_nv_attach_impl
    # pos
    bpftime_llvm_vm
    bpftime_vm
    llvmbpf_vm
  )

else()
  target_include_directories(bpftime-agent
    PRIVATE
    ${FRIDA_GUM_INSTALL_DIR}
    ../include
    ../../third_party/
    ${SPDLOG_INCLUDE}
    ${FRIDA_UPROBE_ATTACH_IMPL_INCLUDE}

  )
  target_link_libraries(bpftime-agent
    ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a
    runtime
    -lpthread
    -lm
    -ldl
    spdlog::spdlog
    bpftime_frida_uprobe_attach_impl
  )
endif()
target_link_options(bpftime-agent PUBLIC "-Wl,-export-dynamic")

if(${BPFTIME_ENABLE_CUDA_ATTACH})
  target_link_options(bpftime-agent PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_nv_attach_impl>" "$<TARGET_FILE:bpftime_llvm_vm>" "$<TARGET_FILE:bpftime_vm>" "$<TARGET_FILE:llvmbpf_vm>" "-Wl,--no-whole-archive")
  target_link_libraries(bpftime-agent PUBLIC bpftime_nv_attach_impl)
  target_include_directories(bpftime-agent PUBLIC ${NV_ATTACH_IMPL_INCLUDE})
endif()
