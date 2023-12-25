add_executable(bpftime-vm-test-runner main.cpp)
set_property(TARGET bpftime-vm-test-runner PROPERTY CXX_STANDARD 20)
add_dependencies(bpftime-vm-test-runner vm-bpf)

target_include_directories(bpftime-vm-test-runner PRIVATE ../include)
target_link_libraries(bpftime-vm-test-runner vm-bpf)
