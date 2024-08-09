add_executable(bpftimetool
  main.cpp
)


if(${BPFTIME_BUILD_WITH_LIBBPF})

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
else()
target_include_directories(bpftimetool PUBLIC
  ../../third_party/
  ../../runtime/include
)
target_link_libraries(bpftimetool
  -lpthread
  -lm
  -ldl
  runtime
)
endif()

install(TARGETS bpftimetool CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)
