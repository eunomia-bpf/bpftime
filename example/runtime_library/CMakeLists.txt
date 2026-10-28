# The example uses uBPF directly; no injected agent or syscall server.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT BPFTIME_UBPF_JIT)
  message(FATAL_ERROR "runtime_library requires Linux and BPFTIME_UBPF_JIT=ON")
endif()
if(BPFTIME_ENABLE_CUDA_ATTACH)
  message(FATAL_ERROR "runtime_library is a CPU example; disable CUDA attach")
endif()

add_executable(runtime_library main.cpp
  $<TARGET_OBJECTS:bpftime_ubpf_vm_obj>)
target_link_libraries(runtime_library PRIVATE
  runtime bpftime_simple_attach_impl)
target_include_directories(runtime_library PRIVATE ${SIMPLE_ATTACH_IMPL_INCLUDE})
# The object above retains constructor-based uBPF factory registration,
# which an ordinary static archive link may discard.

if(BPFTIME_ENABLE_UNIT_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  foreach(mode interpreter jit)
    add_test(NAME runtime_library_${mode}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py
        $<TARGET_FILE:runtime_library> ${mode})
    set_tests_properties(runtime_library_${mode} PROPERTIES TIMEOUT 60)
  endforeach()
endif()
