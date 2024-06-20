add_library(bpftime-agent SHARED
  agent.cpp
)
add_dependencies(bpftime-agent FridaGum spdlog::spdlog bpftime_frida_uprobe_attach_impl bpftime_syscall_trace_attach_impl)
set_target_properties(bpftime-agent PROPERTIES CXX_STANDARD 20 LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/agent.version)
target_link_options(bpftime-agent PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/agent.version)
target_include_directories(bpftime-agent
  PRIVATE
  ${FRIDA_GUM_INSTALL_DIR}
  ../include
  ../../third_party/
  ${SPDLOG_INCLUDE}
  ${FRIDA_UPROBE_ATTACH_IMPL_INCLUDE}
  ${SYSCALL_TRACE_ATTACH_IMPL_INCLUDE}
)
target_link_libraries(bpftime-agent
  ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a
  runtime
  -lpthread
  -lm
  -ldl
  spdlog::spdlog
  bpftime_frida_uprobe_attach_impl
  bpftime_syscall_trace_attach_impl
)
