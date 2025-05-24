add_executable(nv_attach_impl_ptx_test ptx_test.cpp)

option(LLVMBPF_CUDA_PATH "Path where CUDA installs")
message(STATUS "LLVMBPF_CUDA_PATH=${LLVMBPF_CUDA_PATH}")
if(NOT LLVMBPF_CUDA_PATH AND NOT ${LLVMBPF_CUDA_PATH})
  message(FATAL_ERROR "Please set LLVMBPF_CUDA_PATH if you want to build nv_attach_impl_ptx_test, otherwise set LLVMBPF_ENABLE_PTX to NO")
endif()
set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
)
target_link_directories(nv_attach_impl_ptx_test PRIVATE ${LLVMBPF_CUDA_PATH}/lib64 ${LLVMBPF_CUDA_PATH}/lib64/stubs)
target_link_libraries(nv_attach_impl_ptx_test
  PRIVATE
  ${LLVM_LIBS}
  llvmbpf_vm
  cuda
  libnvptxcompiler_static.a
  spdlog::spdlog
  bpftime_nv_attach_impl
)
add_dependencies(nv_attach_impl_ptx_test llvmbpf_vm spdlog::spdlog bpftime_nv_attach_impl)
set(LLVMBPF_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../vm/llvm-jit)
target_include_directories(nv_attach_impl_ptx_test PRIVATE ${LLVM_INCLUDE_DIRS} ${LLVMBPF_ROOT}/src ${LLVMBPF_ROOT}/include ${LLVMBPF_CUDA_PATH}/include ${SPDLOG_INCLUDES} ${NV_ATTACH_IMPL_INCLUDE})
