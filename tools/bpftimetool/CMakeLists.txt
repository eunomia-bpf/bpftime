add_executable(bpftimetool
  main.cpp
)

if(APPLE)
  # Suppress Boost template warning on macOS (Boost/compiler compatibility issue)
  target_compile_options(bpftimetool PRIVATE -Wno-missing-template-arg-list-after-template-kw)
endif()

if(${BPFTIME_ENABLE_CUDA_ATTACH})
  include(../../cmake/cuda.cmake)
  find_cuda()
  target_link_directories(bpftimetool PUBLIC ${CUDA_LIBRARY_PATH})
  target_include_directories(bpftimetool PUBLIC ${CUDA_INCLUDE_PATH})
endif()
if(${BPFTIME_BUILD_WITH_LIBBPF})
  set (LIBBPF_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/../../runtime/object/libbpf)
  target_include_directories(bpftimetool PUBLIC
    ../../third_party/
    ../../runtime/include
    ${LIBBPF_INCLUDE_DIRS}/uapi
    ${LIBBPF_INCLUDE_DIRS}
  )

  if(${BPFTIME_ENABLE_CUDA_ATTACH})
    target_link_libraries(bpftimetool
      -lpthread
      -lm
      -ldl
      runtime
      bpftime-object
      ${CUDA_LIBS}
      bpftime_nv_attach_impl
    )
  else()
    target_link_libraries(bpftimetool
      -lpthread
      -lm
      -ldl
      runtime
      bpftime-object
    )
  endif()
else()
  target_include_directories(bpftimetool PUBLIC
    ../../third_party/
    ../../runtime/include
  )
  if(${BPFTIME_ENABLE_CUDA_ATTACH})
    target_link_libraries(bpftimetool
      -lpthread
      -lm
      -ldl
      runtime
      ${CUDA_LIBS}
      bpftime_nv_attach_impl
    )
  else()
    target_link_libraries(bpftimetool
      -lpthread
      -lm
      -ldl
      runtime
    )
  endif()
endif()

install(TARGETS bpftimetool CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)
