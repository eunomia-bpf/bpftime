# SPDX-License-Identifier: Apache-2.0
#
# Single-process functional test for the no-frida embed trigger. Not installed
# and not part of the default library target; build explicitly with
# `cmake --build <dir> --target inproc_embed_selftest`.
set(BPF_OBJ ${CMAKE_CURRENT_BINARY_DIR}/inproc_embed_selftest.bpf.o)
add_ebpf_program_target(
  inproc_embed_selftest_bpf
  ${CMAKE_CURRENT_SOURCE_DIR}/inproc_embed_selftest.bpf.c
  ${BPF_OBJ})

add_executable(inproc_embed_selftest inproc_embed_selftest.cpp)
add_dependencies(inproc_embed_selftest inproc_embed_selftest_bpf)
target_compile_definitions(inproc_embed_selftest
  PRIVATE BPFTIME_INPROC_EMBED_SELFTEST_BPF_OBJ="${BPF_OBJ}")
target_link_libraries(inproc_embed_selftest PRIVATE runtime bpftime-object
                                                  ${CMAKE_DL_LIBS} rt)
# Anchor trampolines live in this executable; export them to the dynamic symbol
# table so the embed library's dlsym(RTLD_DEFAULT) can resolve them.
target_link_options(inproc_embed_selftest PRIVATE "-Wl,--export-dynamic")
target_include_directories(inproc_embed_selftest PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../object
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
