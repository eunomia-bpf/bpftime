add_library(ptxpass_core STATIC src/core.cpp)
target_include_directories(ptxpass_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/third_party)
set_property(TARGET ptxpass_core PROPERTY CXX_STANDARD 20)
target_link_libraries(ptxpass_core PUBLIC llvmbpf_vm)

if(DEFINED LLVM_INCLUDE_DIRS)
    target_include_directories(ptxpass_core PRIVATE ${LLVM_INCLUDE_DIRS})
endif()

get_target_property(LLVMBPF_VM_INCLUDES llvmbpf_vm INCLUDE_DIRECTORIES)
if(LLVMBPF_VM_INCLUDES)
    target_include_directories(ptxpass_core PRIVATE ${LLVMBPF_VM_INCLUDES})
endif()




