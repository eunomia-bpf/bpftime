# Real-helper bpf_get_stackid/bpf_get_stack benchmark. Only meaningful with
# the Frida-gum uprobe backend, which supplies the real GumBacktracer walk.
if(BPFTIME_ENABLE_FRIDA)
  add_executable(stackid-bench bench_stackid.cpp)
  add_dependencies(stackid-bench runtime bpftime_frida_uprobe_attach_impl)
  target_link_libraries(stackid-bench PRIVATE
    bpftime_frida_uprobe_attach_impl
    runtime
    spdlog::spdlog
    pthread
    ${CMAKE_DL_LIBS})
  target_include_directories(stackid-bench PRIVATE
    ${FRIDA_UPROBE_ATTACH_IMPL_INCLUDE}
    ${BASE_ATTACH_IMPL_INCLUDE}
    ${FRIDA_GUM_INSTALL_DIR}
    ${CMAKE_SOURCE_DIR}/runtime/include)
  set_property(TARGET stackid-bench PROPERTY CXX_STANDARD 20)
endif()
