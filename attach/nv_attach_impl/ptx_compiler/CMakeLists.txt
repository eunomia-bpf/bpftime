add_library(nv_attach_impl_ptx_compiler SHARED ptx_compiler.cpp)


include(../../../cmake/cuda.cmake)
find_cuda()

target_include_directories(nv_attach_impl_ptx_compiler PUBLIC ${CUDA_INCLUDE_PATH})

target_link_directories(nv_attach_impl_ptx_compiler PUBLIC ${CUDA_LIBRARY_PATH} )

target_link_libraries(nv_attach_impl_ptx_compiler PUBLIC  ${CUDA_LIBS} )
set_property(TARGET nv_attach_impl_ptx_compiler PROPERTY CXX_STANDARD 20)
