add_executable(ptxpass_kretprobe main.cpp)
target_include_directories(ptxpass_kretprobe PRIVATE ${CMAKE_SOURCE_DIR}/third_party ${CMAKE_CURRENT_SOURCE_DIR}/../ptxpass_core/include)
target_link_libraries(ptxpass_kretprobe PRIVATE ptxpass_core llvmbpf_vm)
set_property(TARGET ptxpass_kretprobe PROPERTY CXX_STANDARD 20)

# Install/prepare default JSON next to executable in build tree
set(PASS_JSON ${CMAKE_CURRENT_SOURCE_DIR}/kretprobe.json)
if(EXISTS ${PASS_JSON})
  configure_file(${PASS_JSON} ${CMAKE_CURRENT_BINARY_DIR}/ COPYONLY)
endif()



