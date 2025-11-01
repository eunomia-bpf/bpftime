include(../../cmake/cuda.cmake)

add_library(
    bpftime_nv_attach_impl STATIC
    nv_attach_impl.cpp
    nv_attach_private_data.cpp
    nv_attach_impl_frida_setup.cpp
    nv_attach_impl_register_guard_patcher.cpp
    nv_attach_impl_patcher.cpp
    nv_attach_utils.cpp
    ptxpass_pipeline.cpp
)

add_dependencies(bpftime_nv_attach_impl bpftime_base_attach_impl spdlog::spdlog FridaGum  llvmbpf_vm bpftime_vm_compat)
set(NV_ATTACH_IMPL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} )
find_cuda()

target_include_directories(bpftime_nv_attach_impl PRIVATE ${SPDLOG_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR}/third_party PUBLIC ${CMAKE_SOURCE_DIR}/runtime/ ${BASE_ATTACH_IMPL_INCLUDE} ${NV_ATTACH_IMPL_INCLUDE} ${CUDA_INCLUDE_PATH} ${FRIDA_GUM_INSTALL_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime ${CMAKE_CURRENT_SOURCE_DIR}/../../vm/llvm-jit/include ${CMAKE_CURRENT_SOURCE_DIR}/pass/ptxpass_core/include)

target_link_directories(bpftime_nv_attach_impl PUBLIC ${CUDA_LIBRARY_PATH} )

target_link_libraries(bpftime_nv_attach_impl PUBLIC bpftime_base_attach_impl spdlog::spdlog ${CUDA_LIBS} PRIVATE  ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a PUBLIC llvmbpf_vm bpftime_vm_compat)

set_property(TARGET bpftime_nv_attach_impl PROPERTY CXX_STANDARD 20)

target_link_options(bpftime_nv_attach_impl PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_nv_attach_impl>" "-Wl,--no-whole-archive")
# Note: If consumers need whole-archive behavior, they should use:
# -Wl,--whole-archive target -Wl,--no-whole-archive in their link libraries

# PTX transform passes now live under pass/
add_subdirectory(pass/ptxpass_core)
add_subdirectory(pass/ptxpass_kprobe_entry)
add_subdirectory(pass/ptxpass_kretprobe)
add_subdirectory(pass/ptxpass_kprobe_memcapture)

# Copy default pass JSON definitions into build tree
set(PASSES_JSON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pass)
set(PASSES_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/pass)
file(MAKE_DIRECTORY ${PASSES_OUT_DIR})
file(GLOB PASS_JSON_FILES ${PASSES_JSON_DIR}/*.json)
foreach(f ${PASS_JSON_FILES})
  configure_file(${f} ${PASSES_OUT_DIR}/ COPYONLY)
endforeach()

# Provide default passes dir at build time (can be overridden at runtime via env)
target_compile_definitions(bpftime_nv_attach_impl PUBLIC DEFAULT_PASSES_DIR="${PASSES_OUT_DIR}")

add_subdirectory(test)
