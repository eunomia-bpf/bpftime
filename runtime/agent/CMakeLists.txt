add_library(bpftime-agent SHARED
  agent.cpp
)

set(AGENT_DEPS spdlog::spdlog)
if(BPFTIME_ENABLE_FRIDA)
  list(APPEND AGENT_DEPS FridaGum bpftime_frida_uprobe_attach_impl)
endif()
if(${BPFTIME_BUILD_WITH_LIBBPF})
  list(APPEND AGENT_DEPS bpftime_syscall_trace_attach_impl)
endif()
add_dependencies(bpftime-agent ${AGENT_DEPS})

set_target_properties(bpftime-agent PROPERTIES CXX_STANDARD 20 LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/agent.version)

if(UNIX AND NOT APPLE)
  target_link_options(bpftime-agent PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/agent.version)
endif()

if(${TEST_LCOV})
  target_compile_options(bpftime-agent PRIVATE -fprofile-arcs -ftest-coverage -fprofile-update=atomic)
  target_link_options(bpftime-agent PRIVATE -fprofile-arcs)
endif()

# Force link VM compat libraries so constructor-based factory registration is retained
if(${BPFTIME_UBPF_JIT})
  add_dependencies(bpftime-agent bpftime_ubpf_vm)
  target_link_options(bpftime-agent PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_ubpf_vm>" "-Wl,--no-whole-archive")
  target_link_libraries(bpftime-agent PUBLIC bpftime_ubpf_vm)
endif()

if(${BPFTIME_LLVM_JIT})
  add_dependencies(bpftime-agent bpftime_llvm_vm llvmbpf_vm)
  target_link_libraries(bpftime-agent PUBLIC bpftime_llvm_vm)
  # Statically embed the compat_llvm constructor TU to guarantee VM factory registration in LD_PRELOAD context
  target_sources(bpftime-agent PRIVATE $<TARGET_OBJECTS:bpftime_llvm_vm_obj>)
endif()

if(${BPFTIME_ENABLE_CUDA_ATTACH})
  add_dependencies(bpftime-agent bpftime_llvm_vm llvmbpf_vm bpftime_vm)
  target_link_options(bpftime-agent PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_nv_attach_impl>" "$<TARGET_FILE:bpftime_llvm_vm>" "$<TARGET_FILE:bpftime_vm>" "$<TARGET_FILE:llvmbpf_vm>" "-Wl,--no-whole-archive")
  target_link_libraries(bpftime-agent PUBLIC bpftime_nv_attach_impl)
  target_include_directories(bpftime-agent PUBLIC ${NV_ATTACH_IMPL_INCLUDE})
  set(EXTRA_LINK_LIBS bpftime_llvm_vm
    bpftime_vm
    llvmbpf_vm)
endif()

# Include directories
target_include_directories(bpftime-agent
  PRIVATE
  ../include
  ../../third_party/
  ${SPDLOG_INCLUDE}
)
if(BPFTIME_ENABLE_FRIDA)
  target_include_directories(bpftime-agent PRIVATE
    ${FRIDA_GUM_INSTALL_DIR}
    ${FRIDA_UPROBE_ATTACH_IMPL_INCLUDE}
  )
endif()
if(${BPFTIME_BUILD_WITH_LIBBPF})
  target_include_directories(bpftime-agent PRIVATE
    ${SYSCALL_TRACE_ATTACH_IMPL_INCLUDE}
    ${NV_ATTACH_IMPL_INCLUDE}
  )
endif()

# Link libraries
target_link_libraries(bpftime-agent
  PUBLIC
  runtime
  -lpthread
  -lm
  -ldl
  spdlog::spdlog
  ${EXTRA_LINK_LIBS}
)
if(BPFTIME_ENABLE_FRIDA)
  target_link_libraries(bpftime-agent PUBLIC
    ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a
    bpftime_frida_uprobe_attach_impl
  )
  target_compile_definitions(bpftime-agent PRIVATE BPFTIME_HAVE_FRIDA_ATTACH=1)
endif()
if(${BPFTIME_BUILD_WITH_LIBBPF})
  target_link_libraries(bpftime-agent PUBLIC bpftime_syscall_trace_attach_impl)
endif()

if(TARGET bpftime_trap_uprobe_attach_impl)
  add_dependencies(bpftime-agent bpftime_trap_uprobe_attach_impl)
  target_include_directories(bpftime-agent PRIVATE ${TRAP_UPROBE_ATTACH_IMPL_INCLUDE})
  target_link_libraries(bpftime-agent PUBLIC bpftime_trap_uprobe_attach_impl)
  target_compile_definitions(bpftime-agent PRIVATE BPFTIME_HAVE_TRAP_UPROBE_ATTACH=1)
endif()
target_link_options(bpftime-agent PUBLIC "-Wl,-export-dynamic")
