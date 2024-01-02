add_executable(bpftimetool
  main.cpp
)

target_include_directories(bpftimetool PUBLIC
  set(LIBBPF_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/../../runtime/object/libbpf)
  ../../third_party/
  ../../runtime/include
  ${LIBBPF_INCLUDE_DIRS}/uapi
  ${LIBBPF_INCLUDE_DIRS}
)

target_link_libraries(bpftimetool
  -lpthread
  -lm
  -ldl
  runtime
  bpftime-object
)

install(TARGETS bpftimetool CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)
