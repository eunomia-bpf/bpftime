add_library(ptxpass_kprobe_entry SHARED main.cpp)

target_include_directories(
  ptxpass_kprobe_entry 
  PRIVATE 
  ${CMAKE_SOURCE_DIR}/third_party 
  ${CMAKE_CURRENT_SOURCE_DIR}/../ptxpass_core/include
  )

target_link_libraries(ptxpass_kprobe_entry PRIVATE ptxpass_core llvmbpf_vm)
add_dependencies(ptxpass_kprobe_entry ptxpass_core llvmbpf_vm)
set_property(TARGET ptxpass_kprobe_entry PROPERTY CXX_STANDARD 20)
add_dependencies(ptxpass_kprobe_entry ptxpass_core llvmbpf_vm)


