project(
    "bpftime-syscall-polyfill"
    VERSION 0.1.0
    LANGUAGES C CXX
)

add_library(bpftime-syscall-server SHARED
server.cpp)
target_link_libraries(bpftime-syscall-server PUBLIC
runtime
-lpthread
-lm 
-ldl)
target_include_directories(bpftime-syscall-server  
    PUBLIC 
    "../../core" 
    "../../third_party/libbpf/include" 
    "../../third_party/libbpf/include/uapi"
)
