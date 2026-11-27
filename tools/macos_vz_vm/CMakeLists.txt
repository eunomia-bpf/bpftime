# SPDX-License-Identifier: MIT
#
# bpftime-macos-exec is a tiny Linux-only binfmt_misc interpreter that runs
# inside the ARM64 guest. The Swift bpftime-vz manager is compiled on the
# macOS host directly with swiftc and intentionally has no CMake target here.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(bpftime-macos-exec bpftime_macos_exec.c)
  target_compile_options(bpftime-macos-exec PRIVATE -O2 -Wall -Wextra)
  install(TARGETS bpftime-macos-exec RUNTIME DESTINATION bin)
  install(PROGRAMS register_binfmt.sh DESTINATION bin)
  install(PROGRAMS vm_exec.sh DESTINATION bin)
endif()
