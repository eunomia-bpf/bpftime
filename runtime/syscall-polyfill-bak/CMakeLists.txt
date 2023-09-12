project(
    "bpftime-syscall-polyfill"
    VERSION 0.1.0
    LANGUAGES C CXX
)

function(polyfill_setup_target target out_name)
    set_target_properties(${target} PROPERTIES CXX_STANDARD 20 PREFIX "" OUTPUT_NAME ${out_name})
    target_include_directories(${target} PUBLIC ${LOCAL_INCLUDES})

    # target_include_directories(${target} PUBLIC ${FRIDA_CORE_INSTALL_DIR} ${FRIDA_GUM_INSTALL_DIR})
endfunction()

# agent.so
add_library(bpf-syscall-agent SHARED
agent.cpp 
helpers_impl.cpp
wrapper.cpp
task_struct_wrapper.c)
target_link_libraries(bpf-syscall-agent  
core-bpf
dl 
pthread)
target_include_directories(bpf-syscall-agent  
    PUBLIC 
    "../../core" 
    "../../third_party/libbpf/include" 
    "../../third_party/libbpf/include/uapi"
)
