add_library(bpftime_vm_compat ./src/ebpf_vm_compat.c)

target_include_directories(bpftime_vm_compat INTERFACE ./include PRIVATE ./include)
