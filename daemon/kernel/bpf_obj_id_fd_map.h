/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
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

SEC("kprobe/security_bpf_map")
int BPF_KPROBE(security_bpf_map_enter, struct bpf_map *map)
{
	u64 pid_tgid = bpf_get_current_pid_tgid();
	u32 id = BPF_CORE_READ(map, id);
	bpf_printk("security_bpf_map enter, id %d", id);
	bpf_map_update_elem(&bpf_map_new_fd_args_map, &pid_tgid, &id, 0);
	return 0;
}

// key is id
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u64);
	__type(value, struct bpf_insn_data);
} bpf_prog_insns_map SEC(".maps");

struct bpf_insn_data insns_data = { 0 };

// key can be either id or pid_tgid
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u64);
	__type(value, u32);
} bpf_progs_new_fd_args_map SEC(".maps");

SEC("kprobe/bpf_prog_kallsyms_add")
int BPF_KPROBE(bpf_prog_kallsyms_add_enter, struct bpf_prog *prog)
{
	u64 pid_tgid = bpf_get_current_pid_tgid();
	u32 id = BPF_CORE_READ(prog, aux, id);
	bpf_printk("bpf_prog_kallsyms_add enter find id: %d", id);
	bpf_map_update_elem(&bpf_progs_new_fd_args_map, &pid_tgid, &id, 0);

	struct bpf_insn_data *data = (struct bpf_insn_data *)bpf_map_lookup_elem(
		&bpf_prog_insns_map, &pid_tgid);
	if (!data) {
		bpf_printk(
			"bpf_prog_kallsyms_add enter find id: %d, but no insns",
			id);
		return 0;
	}
	u64 id_key = id;
	long res = bpf_map_update_elem(&bpf_prog_insns_map, &id_key, data, 0);
	if (res) {
		bpf_printk(
			"bpf_prog_kallsyms_add enter find id: %d, but update failed",
			id);
		return 0;
	}
	bpf_map_delete_elem(&bpf_prog_insns_map, &pid_tgid);
	return 0;
}

#endif // BPFTIME_BPF_MAPS_ID_FD_MAP
