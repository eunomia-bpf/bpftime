add_executable(bpftimetool
  main.cpp
)
if(${BPFTIME_ENABLE_CUDA_ATTACH})
  include(../../cmake/cuda.cmake)
  find_cuda()
  target_link_directories(bpftimetool PUBLIC ${CUDA_LIBRARY_PATH})
  target_include_directories(bpftimetool PUBLIC ${CUDA_INCLUDE_PATH})
endif()
if(${BPFTIME_BUILD_WITH_LIBBPF})
  target_include_directories(bpftimetool PUBLIC
    set (LIBBPF_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/../../runtime/object/libbpf)
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
    ${CUDA_LIBS}
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
    ${CUDA_LIBS}
  )
endif()

install(TARGETS bpftimetool CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)
