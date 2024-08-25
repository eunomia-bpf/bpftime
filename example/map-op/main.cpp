#include <iostream>
#include <unistd.h>
#if __linux__
#include <linux/bpf.h>
#include <sys/syscall.h>
#elif defined(__APPLE__)
#include "bpftime_epoll.h"
#endif


#if defined(__APPLE__)
using namespace bpftime_epoll;
#endif

static struct bpf_insn prog[] = {
    BPF_MOV64_IMM(BPF_REG_0, 0),
    BPF_EXIT_INSN(),
};

static long bpf(int cmd, union bpf_attr *attr, unsigned int size) {
    #if __linux__
    return syscall(__NR_bpf, cmd, attr, size);
    #endif
    return 0;
}

int main() {
    union bpf_attr attr = {};
    int prog_fd, map_fd;

    attr.prog_type = BPF_PROG_TYPE_SOCKET_FILTER;
    attr.insns = (unsigned long) prog;
    attr.insn_cnt = sizeof(prog) / sizeof(prog[0]);
    attr.license = (unsigned long) "GPL";

    prog_fd = bpf(BPF_PROG_LOAD, &attr, sizeof(attr));
    if (prog_fd < 0) {
        std::cerr<<"Failed to load eBPF program";
        return 1;
    }
    printf("eBPF program loaded and fd: %d\n",prog_fd);

    attr = (union bpf_attr){0};
    attr.map_type = BPF_MAP_TYPE_ARRAY;
    attr.key_size = sizeof(int);
    attr.value_size = sizeof(int);
    attr.max_entries = 1;

    map_fd = bpf(BPF_MAP_CREATE, &attr, sizeof(attr));
    if(map_fd < 0) {
        std::cerr<<"Failed to create eBPF map";
        return 1;
    }
    printf("eBPF map created with fd:%d\n", map_fd);
    int key = 0, value = 42;

    attr = (union bpf_attr){0};
    attr.map_fd = map_fd;
    attr.key = (unsigned long)&key;
    attr.value = (unsigned long)&value;
    attr.flags = BPF_XDP;

    if(bpf(BPF_MAP_UPDATE_ELEM, &attr, sizeof(attr)) < 0) {
        std::cerr<<"Error in updating eBPF map\n";
        return 1;
    }
    printf("Map updated successfully\n");

    close(prog_fd);
    close(map_fd);
    return 0;
}