include(../../cmake/cuda.cmake)

add_library(
    bpftime_nv_attach_impl STATIC
    nv_attach_impl.cpp
    nv_attach_private_data.cpp
    cuda_injector.cpp
    nv_attach_impl_frida_setup.cpp
    nv_attach_impl_patcher.cpp
    nv_attach_impl_register_guard_patcher.cpp
)

add_dependencies(bpftime_nv_attach_impl bpftime_base_attach_impl spdlog::spdlog FridaGum pos llvmbpf_vm bpftime_vm_compat)
set(NV_ATTACH_IMPL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} )
find_cuda()

target_include_directories(bpftime_nv_attach_impl PRIVATE ${SPDLOG_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR} PUBLIC ${CMAKE_SOURCE_DIR}/runtime/ ${BASE_ATTACH_IMPL_INCLUDE} ${NV_ATTACH_IMPL_INCLUDE} ${CUDA_INCLUDE_PATH} ${FRIDA_GUM_INSTALL_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime ${CMAKE_CURRENT_SOURCE_DIR}/../../vm/llvm-jit/include)

target_link_directories(bpftime_nv_attach_impl PUBLIC ${CUDA_LIBRARY_PATH} pos)

target_link_libraries(bpftime_nv_attach_impl PUBLIC bpftime_base_attach_impl spdlog::spdlog ${CUDA_LIBS} PRIVATE pos ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a PUBLIC llvmbpf_vm bpftime_vm_compat)

set_property(TARGET bpftime_nv_attach_impl PROPERTY CXX_STANDARD 20)

target_link_options(bpftime_nv_attach_impl PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_nv_attach_impl>" "-Wl,--no-whole-archive")

add_subdirectory(test)
