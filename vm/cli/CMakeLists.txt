add_executable(bpftime-simple-cli simple-cli.cpp)
set_property(TARGET bpftime-simple-cli PROPERTY CXX_STANDARD 20)
add_dependencies(bpftime-simple-cli vm-bpf)

target_include_directories(bpftime-simple-cli PRIVATE ../include)
target_link_libraries(bpftime-simple-cli vm-bpf)
