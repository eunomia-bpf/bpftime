// SPDX-License-Identifier: MIT
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include "bpf_test_println.h"

// Slots 0/1 count entry/return waves through the lookup result. Slots 2/3
// demonstrate CPU-serviced updates. Slot 4 counts unexpected helper results.
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 5);
	__type(key, __u32);
	__type(value, __u64);
} results SEC(".maps");

// CPU owns slot 0 (request); entry and return probes own slots 1 and 2.
// Echo a function of the full request, so old responses cannot complete a
// new round. Normal entry/exit modes do not run these additional probes.
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 3);
	__type(key, __u32);
	__type(value, __u64);
} mailbox SEC(".maps");

SEC("kprobe/rocm__bpftime_example_kernel")
int rt_entry__rocm(void *ctx)
{
	__u32 key = 0;
	__u64 *request = bpf_map_lookup_elem(&mailbox, &key);
	if (!request)
		return 0;
	__u64 input = *request;
	if (!input)
		return 0;
	__u64 reply = input * 3 + 1;
	key = 1;
	bpf_map_update_elem(&mailbox, &key, &reply, BPF_ANY);
	return 0;
}

SEC("kretprobe/rocm__bpftime_example_kernel")
int rt_exit__rocm(void *ctx)
{
	__u32 key = 0;
	__u64 *request = bpf_map_lookup_elem(&mailbox, &key);
	if (!request)
		return 0;
	__u64 input = *request;
	if (!input)
		return 0;
	key = 2;
	__u64 *reply = bpf_map_lookup_elem(&mailbox, &key);
	if (reply)
		// Multiple waves may respond concurrently. This writes the real
		// mapped value on the GPU, without an update helper round trip.
		__sync_lock_test_and_set(reply, input * 7 + 2);
	return 0;
}

static __always_inline void record(__u32 key)
{
	__u64 *count = bpf_map_lookup_elem(&results, &key);
	if (count)
		__sync_fetch_and_add(count, 1);
	__u32 updated = key + 2;
	__u64 value = 123 + key;
	int failed =
		bpf_map_update_elem(&results, &updated, &value, BPF_ANY) != 0;
	// Array slots always exist, and cannot be deleted.
	failed |= bpf_map_update_elem(&results, &updated, &value,
				      BPF_NOEXIST) != -17;
	failed |= bpf_map_delete_elem(&results, &updated) != -22;
	__u32 missing = 5;
	failed |= bpf_map_lookup_elem(&results, &missing) != 0;
	failed |=
		bpf_map_update_elem(&results, &missing, &value, BPF_ANY) != -7;
	if (failed) {
		__u32 errors = 4;
		__u64 *error_count = bpf_map_lookup_elem(&results, &errors);
		if (error_count)
			__sync_fetch_and_add(error_count, 1);
	}
}

SEC("kprobe/rocm__bpftime_example_kernel")
int entry__rocm(void *ctx)
{
	record(0);
	char message[] = "BPF entry";
	bpf_test_println(message);
	return 0;
}

SEC("kretprobe/rocm__bpftime_example_kernel")
int exit__rocm(void *ctx)
{
	record(1);
	char message[] = "BPF exit";
	bpf_test_println(message);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
