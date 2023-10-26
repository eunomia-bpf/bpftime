#ifndef BPFTIME_BPF_MAPS_ID_FD_MAP
#define BPFTIME_BPF_MAPS_ID_FD_MAP

#include <vmlinux.h>
#include "bpf_defs.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

// pid & fd for map ids
// struct {
// 	__uint(type, BPF_MAP_TYPE_HASH);
// 	__uint(max_entries, 10240);
// 	__type(key, u64);
// 	__type(value, u32);
// } bpf_map_pid_fd_id_map SEC(".maps");

// #define PID_MASK_FOR_PFD 0xffffffff00000000
// #define FD_MASK_FOR_PFD 0x00000000ffffffff
// #define MAKE_PFD(pid, fd) (((u64)pid << 32) | (u64)fd)

// static __always_inline void set_pid_fd_with_map_id(u32 fd, u32 id) {
// 	if (fd < 0 || id < 0) {
// 		return;
// 	}
// 	u32 pid = bpf_get_current_pid_tgid() >> 32;
// 	u64 key = MAKE_PFD(pid, fd);
// 	bpf_map_update_elem(&bpf_map_pid_fd_id_map, &key, &id, 0);
// }

// static __always_inline int get_map_id_with_pid_fd(u32 fd) {
//     u32 pid = bpf_get_current_pid_tgid() >> 32;
// 	u64 key = MAKE_PFD(pid, fd);
//     int* id_ptr = bpf_map_lookup_elem(&bpf_map_pid_fd_id_map, &key);
//     if (!id_ptr) {
//         return -1;
//     }
//     return (int)*id_ptr;
// }

// static __always_inline void remove_pid_fd_with_map_id(u32 fd) {
// 	u32 pid = bpf_get_current_pid_tgid() >> 32;
// 	u64 key = MAKE_PFD(pid, fd);
// 	bpf_map_delete_elem(&bpf_map_pid_fd_id_map, &key);
// }

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
	u32 id = BPF_CORE_READ(prog, id);
	bpf_map_update_elem(&bpf_progs_new_fd_args_map, &pid_tgid, &prog, 0);
	return 0;
}

#endif // BPFTIME_BPF_MAPS_ID_FD_MAP
