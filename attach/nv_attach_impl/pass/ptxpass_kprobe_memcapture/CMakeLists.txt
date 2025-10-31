add_executable(ptxpass_kprobe_memcapture main.cpp)
target_include_directories(ptxpass_kprobe_memcapture PRIVATE ${CMAKE_SOURCE_DIR}/third_party ${CMAKE_CURRENT_SOURCE_DIR}/../ptxpass_core/include)
target_link_libraries(ptxpass_kprobe_memcapture PRIVATE ptxpass_core llvmbpf_vm)
add_dependencies(ptxpass_kprobe_memcapture ptxpass_core llvmbpf_vm)
set_property(TARGET ptxpass_kprobe_memcapture PROPERTY CXX_STANDARD 20)

# Install/prepare default JSON next to executable in build tree
set(PASS_JSON ${CMAKE_CURRENT_SOURCE_DIR}/kprobe_memcapture.json)
if(EXISTS ${PASS_JSON})
  configure_file(${PASS_JSON} ${CMAKE_CURRENT_BINARY_DIR}/ COPYONLY)
endif()



