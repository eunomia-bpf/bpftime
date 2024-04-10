add_executable(bpftime-vm-test-runner main.cpp)
set_property(TARGET bpftime-vm-test-runner PROPERTY CXX_STANDARD 20)
add_dependencies(bpftime-vm-test-runner bpftime_vm_compat)

target_include_directories(bpftime-vm-test-runner PRIVATE  ${SPDLOG_INCLUDE})
target_link_libraries(bpftime-vm-test-runner bpftime_vm_compat spdlog::spdlog)
