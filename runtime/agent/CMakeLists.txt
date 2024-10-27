add_library(bpftime-agent SHARED
  agent.cpp
)
if(${BPFTIME_BUILD_WITH_LIBBPF})
  add_dependencies(bpftime-agent FridaGum spdlog::spdlog bpftime_frida_uprobe_attach_impl bpftime_syscall_trace_attach_impl)
else()
  add_dependencies(bpftime-agent FridaGum spdlog::spdlog bpftime_frida_uprobe_attach_impl)
endif()
set_target_properties(bpftime-agent PROPERTIES CXX_STANDARD 20 LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/agent.version)
if(UNIX AND NOT APPLE)
target_link_options(bpftime-agent PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/agent.version)
endif()

if (${TEST_LCOV}) 
  target_compile_options(bpftime-agent PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
  target_link_options(bpftime-agent PRIVATE -fprofile-arcs)
endif()

if(${BPFTIME_BUILD_WITH_LIBBPF})
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
