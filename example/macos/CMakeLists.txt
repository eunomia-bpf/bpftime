# Cross-platform userspace uprobe + eBPF map demo. The Frida-gum attach
# implementation is required on both macOS (the primary target) and Linux.
if(NOT BPFTIME_ENABLE_FRIDA)
  message(FATAL_ERROR
    "the macos example requires -DBPFTIME_ENABLE_FRIDA=ON")
endif()

add_library(bpftime_macos_demo_target SHARED target.cpp)
# SHARED libraries are position-independent by default in CMake.
set_target_properties(bpftime_macos_demo_target PROPERTIES CXX_STANDARD 20)

add_executable(macos_uprobe_demo main.cpp
  $<TARGET_OBJECTS:bpftime_ubpf_vm_obj>)
target_include_directories(macos_uprobe_demo PRIVATE
  ${FRIDA_UPROBE_ATTACH_IMPL_INCLUDE}
  ${BASE_ATTACH_IMPL_INCLUDE})
# Retain constructor-based uBPF factory registration, mirroring
# runtime_library. The Frida impl carries the Darwin framework/libraries;
# librt (for shm_open on older glibc) arrives transitively from runtime.
target_link_libraries(macos_uprobe_demo PRIVATE
  runtime
  bpftime_frida_uprobe_attach_impl
  bpftime_macos_demo_target)
set_target_properties(macos_uprobe_demo PROPERTIES
  CXX_STANDARD 20
  BUILD_RPATH "$<TARGET_FILE_DIR:macos_uprobe_demo>")

if(BPFTIME_ENABLE_UNIT_TESTING)
  foreach(mode interpreter jit)
    add_test(NAME macos_uprobe_demo_${mode}
      COMMAND macos_uprobe_demo ${mode})
    set_tests_properties(macos_uprobe_demo_${mode} PROPERTIES TIMEOUT 60)
  endforeach()
endif()
