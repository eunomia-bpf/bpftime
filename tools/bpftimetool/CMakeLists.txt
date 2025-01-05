add_executable(bpftimetool
  main.cpp
)

include(../../cmake/cuda.cmake)
find_cuda()

if(${BPFTIME_BUILD_WITH_LIBBPF})
  target_include_directories(bpftimetool PUBLIC
    set(LIBBPF_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/../../runtime/object/libbpf)
    ../../third_party/
    ../../runtime/include
    ${LIBBPF_INCLUDE_DIRS}/uapi
    ${LIBBPF_INCLUDE_DIRS}
  )
  target_link_directories(bpftimetool PUBLIC ${CUDA_LIBRARY_PATH})
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
    cuda
    cudart
  )
endif()

install(TARGETS bpftimetool CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)
