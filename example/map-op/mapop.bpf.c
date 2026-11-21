
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <linux/types.h>


struct bpf_map_def {
    unsigned int type;
    unsigned int key_size;
    unsigned int value_size;
    unsigned int max_entries;
};

struct bpf_map_def exec_count __attribute__((section(".maps"))) = {
    .type = BPF_MAP_TYPE_HASH,
    .key_size = sizeof(uint32_t),
    .value_size = sizeof(uint64_t),
    .max_entries = 1024,
};

__attribute__((section("tracepoint/syscalls/sys_enter_execve")))
int count_execve(void *ctx) {
    uint32_t pid = bpf_get_current_pid_tgid() >> 32;
    uint64_t *count = bpf_map_lookup_elem(&exec_count, &pid);
    if (count) {
        (*count)++;
    } else {
        uint64_t initial_count = 1;
        bpf_map_update_elem(&exec_count, &pid, &initial_count, BPF_ANY);
    }
    return 0;
}

char LICENSE[] __attribute__((section("license"))) = "GPL";