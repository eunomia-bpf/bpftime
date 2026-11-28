# bpftime-remote-host: cross-platform target daemon (Linux and macOS).
add_executable(bpftime-remote-host
  main.cpp
  remote_host.cpp
)

set(remote_host_includes
  ${CMAKE_SOURCE_DIR}/runtime/include
  ${CMAKE_SOURCE_DIR}/runtime/src
  ${CMAKE_SOURCE_DIR}/vm/vm-core/include
  ${CMAKE_SOURCE_DIR}/third_party
  ${SPDLOG_INCLUDE}
)

if(APPLE)
  target_sources(bpftime-remote-host PRIVATE
    darwin_target.cpp
    macho_reader.cpp
    elf_synth.cpp
  )
  target_compile_definitions(bpftime-remote-host PRIVATE BPFTIME_HAVE_FRIDA_GUM=1)
  list(APPEND remote_host_includes
    ${CMAKE_SOURCE_DIR}/compat/darwin/include
    ${FRIDA_GUM_INSTALL_DIR}
  )
else()
  target_sources(bpftime-remote-host PRIVATE linux_target.cpp)
endif()

target_include_directories(bpftime-remote-host PRIVATE ${remote_host_includes})
set_target_properties(bpftime-remote-host PROPERTIES CXX_STANDARD 20)
target_link_libraries(bpftime-remote-host PRIVATE runtime)

if(APPLE)
  target_link_libraries(bpftime-remote-host PRIVATE
    ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a
    "-framework Foundation"
    "-framework CoreFoundation"
    "-framework Security"
    "-framework JavaScriptCore"
  )
  add_dependencies(bpftime-remote-host runtime FridaGum spdlog::spdlog)
endif()

# Native end-to-end victim workload (exports a noinline compute()).
add_executable(bpftime_remote_e2e_victim test/victim.c)
set_target_properties(bpftime_remote_e2e_victim PROPERTIES
  C_STANDARD 99
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test)

if(BPFTIME_ENABLE_UNIT_TESTING)
  add_test(
    NAME bpftime_remote_host_e2e
    COMMAND ${CMAKE_COMMAND} -E env
      BPFTIME_REMOTE_PORT=49523
      BPFTIME_REMOTE_HOST_BIN=$<TARGET_FILE:bpftime-remote-host>
      BPFTIME_REMOTE_E2E_VICTIM=$<TARGET_FILE:bpftime_remote_e2e_victim>
      BPFTIME_AGENT_LIB=$<TARGET_FILE:bpftime-agent>
      python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/e2e_remote_host.py)
  set_tests_properties(bpftime_remote_host_e2e PROPERTIES TIMEOUT 180)
endif()
