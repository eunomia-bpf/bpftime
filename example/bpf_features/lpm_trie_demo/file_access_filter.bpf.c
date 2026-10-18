// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
/* bpftime LPM Trie file access control */

#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// LPM Trie key structure (matches kernel's bpf_lpm_trie_key)
struct lpm_key {
	u32 prefixlen; // Prefix length in bits
	char data[64]; // Path data
};

struct event_data {
	u64 timestamp;
	u32 pid;
	u32 tid;
	u32 counter;
	u32 function_id;
	s32 flags;
	char filename[64];
	char comm[16];
	u32 allowed; // 0 = denied, 1 = allowed
};

// LPM Trie map - stores allowed file path prefixes
struct {
	__uint(type, BPF_MAP_TYPE_LPM_TRIE);
	__uint(max_entries, 100);
	__type(key, struct lpm_key);
	__type(value, u32);
	__uint(map_flags, BPF_F_NO_PREALLOC);
} allowed_paths SEC(".maps");

// Queue map for FIFO event processing
struct {
	__uint(type, BPF_MAP_TYPE_QUEUE);
	__uint(max_entries, 64);
	__type(value, struct event_data);
} events_queue SEC(".maps");

// Counter for function calls
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u32);
} call_counter SEC(".maps");

// Check if path matches any allowed prefix using LPM Trie
// bpftime's userspace LPM Trie implementation handles the longest prefix
// matching
static __always_inline int is_path_allowed(const char *path)
{
	struct lpm_key key = {};
	int path_len = 0;

	// Copy path to key data and calculate length
	for (int i = 0; i < 63 && path[i] != '\0'; i++) {
		key.data[i] = path[i];
		path_len++;
	}
	key.data[path_len] = '\0';

	// Set prefix length to maximum possible for this path
	// bpftime's LPM Trie implementation will find the longest matching
	// prefix
	key.prefixlen = path_len * 8;

	// Single lookup - bpftime's userspace LPM Trie handles the matching
	u32 *policy = bpf_map_lookup_elem(&allowed_paths, &key);

	return policy ? *policy : 0; // 0 = denied, 1 = allowed
}

SEC("uprobe/./file_access_target:test_file_access")
int uprobe_test_file_access(struct pt_regs *ctx)
{
	struct event_data event = {};

	// Get function arguments
	char *filename = (char *)PT_REGS_PARM1(ctx);
	s32 flags = (s32)PT_REGS_PARM2(ctx);

	// Fill event data
	event.timestamp = bpf_ktime_get_ns();
	event.pid = bpf_get_current_pid_tgid() >> 32;
	event.tid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
	event.flags = flags;
	event.function_id = 1;

	// Copy filename safely
	if (filename) {
		bpf_probe_read_user_str(event.filename, sizeof(event.filename),
					filename);
	}

	// Get process name
	bpf_get_current_comm(event.comm, sizeof(event.comm));

	// Check if path is allowed using LPM Trie
	event.allowed = is_path_allowed(event.filename);

	// Update counter
	u32 key = 0;
	u32 *counter = bpf_map_lookup_elem(&call_counter, &key);
	if (counter) {
		u32 new_val = *counter + 1;
		bpf_map_update_elem(&call_counter, &key, &new_val, BPF_ANY);
		event.counter = new_val;
	} else {
		u32 init_val = 1;
		bpf_map_update_elem(&call_counter, &key, &init_val, BPF_ANY);
		event.counter = 1;
	}

	// Push event to queue
	bpf_map_push_elem(&events_queue, &event, BPF_ANY);

	// Print debug info
	bpf_printk("LPM Trie: file=%s, allowed=%d, counter=%d", event.filename,
		   event.allowed, event.counter);

	return 0;
}

char _license[] SEC("license") = "GPL";