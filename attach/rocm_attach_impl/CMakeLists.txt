include(../../cmake/rocm.cmake)

add_library(
    bpftime_rocm_attach_impl STATIC
    rocm_attach_impl.cpp
)

add_dependencies(bpftime_rocm_attach_impl bpftime_base_attach_impl spdlog::spdlog FridaGum  llvmbpf_vm bpftime_vm_compat)
set(ROCM_ATTACH_IMPL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} )
find_rocm()

target_include_directories(bpftime_rocm_attach_impl PRIVATE ${SPDLOG_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR} PUBLIC ${CMAKE_SOURCE_DIR}/runtime/ ${BASE_ATTACH_IMPL_INCLUDE} ${ROCM_ATTACH_IMPL_INCLUDE} ${ROCM_INCLUDE_PATH} ${FRIDA_GUM_INSTALL_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime ${CMAKE_CURRENT_SOURCE_DIR}/../../vm/llvm-jit/include)

target_link_directories(bpftime_rocm_attach_impl PUBLIC ${ROCM_LIBRARY_PATH} )

target_link_libraries(bpftime_rocm_attach_impl PUBLIC bpftime_base_attach_impl spdlog::spdlog ${ROCM_LIBS} PRIVATE  ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a PUBLIC llvmbpf_vm bpftime_vm_compat)

set_property(TARGET bpftime_rocm_attach_impl PROPERTY CXX_STANDARD 20)

target_link_options(bpftime_rocm_attach_impl PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_rocm_attach_impl>" "-Wl,--no-whole-archive")

# add_subdirectory(test)
