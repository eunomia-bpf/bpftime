# bpftime-darwin-host: macOS host bridge daemon (Darwin only).
if(NOT APPLE OR NOT BPFTIME_ENABLE_FRIDA)
  return()
endif()

add_executable(bpftime-darwin-host
  main.cpp
  darwin_host.cpp
  macho_reader.cpp
  elf_synth.cpp
)

set_target_properties(bpftime-darwin-host PROPERTIES CXX_STANDARD 20)

target_compile_definitions(bpftime-darwin-host PRIVATE BPFTIME_HAVE_FRIDA_GUM=1)

target_include_directories(bpftime-darwin-host PRIVATE
  ${CMAKE_SOURCE_DIR}/compat/darwin/include
  ${CMAKE_SOURCE_DIR}/runtime/include
  ${CMAKE_SOURCE_DIR}/runtime/src
  ${CMAKE_SOURCE_DIR}/vm/vm-core/include
  ${CMAKE_SOURCE_DIR}/vm/compat/include
  ${CMAKE_SOURCE_DIR}/third_party
  ${SPDLOG_INCLUDE}
  ${FRIDA_GUM_INSTALL_DIR}
)

target_link_libraries(bpftime-darwin-host PRIVATE
  runtime
  spdlog::spdlog
  ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a
  "-framework Foundation"
  "-framework CoreFoundation"
  "-framework Security"
  "-framework JavaScriptCore"
)

add_dependencies(bpftime-darwin-host runtime FridaGum spdlog::spdlog)

# In-process producer for the native event-stream test: opens the host's SHM
# segment and emits ringbuf / software-perf records through the public
# runtime APIs. The standalone daemon repo has no preload agent yet, so the
# test drives production the same way an agent process would.
add_executable(bpftime_darwin_event_producer test/event_producer.cpp)
set_target_properties(bpftime_darwin_event_producer PROPERTIES
  CXX_STANDARD 20
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test)
target_include_directories(bpftime_darwin_event_producer PRIVATE
  ${CMAKE_SOURCE_DIR}/compat/darwin/include
  ${CMAKE_SOURCE_DIR}/runtime/include
  ${CMAKE_SOURCE_DIR}/runtime/src
  ${SPDLOG_INCLUDE}
)
target_link_libraries(bpftime_darwin_event_producer PRIVATE runtime)
add_dependencies(bpftime_darwin_event_producer runtime)

add_executable(bpftime_darwin_e2e_victim test/victim.c)
set_target_properties(bpftime_darwin_e2e_victim PROPERTIES
  C_STANDARD 99
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test)

if(BPFTIME_ENABLE_UNIT_TESTING)
  add_test(
    NAME bpftime_darwin_host_e2e
    COMMAND ${CMAKE_COMMAND} -E env
      BPFTIME_DARWIN_PORT=49512
      BPFTIME_DARWIN_HOST_BIN=$<TARGET_FILE:bpftime-darwin-host>
      BPFTIME_E2E_VICTIM=$<TARGET_FILE:bpftime_darwin_e2e_victim>
      BPFTIME_EVENT_PRODUCER=$<TARGET_FILE:bpftime_darwin_event_producer>
      python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/e2e_darwin_bridge.py)
  set_tests_properties(bpftime_darwin_host_e2e PROPERTIES TIMEOUT 180)
endif()
