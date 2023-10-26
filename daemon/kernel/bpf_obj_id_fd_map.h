#ifndef BPFTIME_BPF_MAPS_ID_FD_MAP
#define BPFTIME_BPF_MAPS_ID_FD_MAP

#include <vmlinux.h>
#include "bpf_defs.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u64);
	__type(value, u32);
} bpf_map_new_fd_args_map SEC(".maps");

SEC("kprobe/bpf_map_init_from_attr")
int BPF_KPROBE(bpf_map_init_from_attr_enter, struct bpf_map *map, int flags)
{	
	bpf_printk("bpf_map_new_fd enter");
	u64 pid_tgid = bpf_get_current_pid_tgid();
	u32 id = BPF_CORE_READ(map, id);
	bpf_map_update_elem(&bpf_map_new_fd_args_map, &pid_tgid, &map, 0);
	return 0;
}

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u64);
	__type(value, u32);
} bpf_progs_new_fd_args_map SEC(".maps");

SEC("kprobe/bpf_prog_kallsyms_add")
int BPF_KPROBE(bpf_prog_kallsyms_add_enter, struct bpf_prog *prog)
{	
	bpf_printk("bpf_prog_kallsyms_add enter");
	u64 pid_tgid = bpf_get_current_pid_tgid();
	u32 id = BPF_CORE_READ(prog, aux, id);
	bpf_map_update_elem(&bpf_progs_new_fd_args_map, &pid_tgid, &prog, 0);
	return 0;
}

#endif // BPFTIME_BPF_MAPS_ID_FD_MAP
