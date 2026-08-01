// SPDX-License-Identifier: GPL-2.0
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 64);
	__type(key, u32);
	__type(value, u64);
} launch_count SEC(".maps");

static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_device_ordinal)(void) = (void *)512;

SEC("kprobe/_Z9vectorAddPKfS0_Pfi")
int count_vector_add_launches(void)
{
	u64 x, y, z;
	bpf_get_block_idx(&x, &y, &z);
	if (x != 0)
		return 0;

	u32 device = (u32)bpf_get_device_ordinal();
	u64 one = 1;
	u64 *count = bpf_map_lookup_elem(&launch_count, &device);
	if (count)
		__atomic_add_fetch(count, 1, __ATOMIC_SEQ_CST);
	else
		bpf_map_update_elem(&launch_count, &device, &one, BPF_NOEXIST);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
