#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// Map to store entry timestamps
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u32);
	__type(value, u64);
} start_ts SEC(".maps");

// Map to store total execution time
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u32);
	__type(value, u64);
} total_time_ns SEC(".maps");

// Map to store call count
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u32);
	__type(value, u64);
} call_count SEC(".maps");

static const void (*ebpf_puts)(const char *) = (void *)501;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int probe()
{
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 ts = bpf_get_globaltimer();
	u64 x, y, z;
	bpf_get_block_idx(&x, &y, &z);
	// Store entry timestamp
	bpf_map_update_elem(&start_ts, &x, &ts, BPF_ANY);

	// Increment call count
	u64 one = 1;
	u64 *cnt = bpf_map_lookup_elem(&call_count, &pid);
	if (cnt) {
		*cnt += 1;
		bpf_map_update_elem(&call_count, &pid, cnt, BPF_EXIST);
	} else {
		bpf_map_update_elem(&call_count, &pid, &one, BPF_NOEXIST);
	}

	bpf_printk("Entered _Z9vectorAddPKfS0_Pf x=%lu, ts=%lu\n", x, ts);

	return 0;
}

SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int retprobe()
{
	u64 x, y, z;
	bpf_get_block_idx(&x, &y, &z);
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 *tsp = bpf_map_lookup_elem(&start_ts, &x);

	if (tsp) {
		u64 delta = bpf_get_globaltimer() - *tsp;
		bpf_map_delete_elem(&start_ts, &pid);

		// Update total time
		u64 *total = bpf_map_lookup_elem(&total_time_ns, &pid);
		if (total) {
			*total += delta;
			bpf_map_update_elem(&total_time_ns, &pid, total,
					    BPF_EXIST);
		} else {
			bpf_map_update_elem(&total_time_ns, &pid, &delta,
					    BPF_NOEXIST);
		}
		bpf_printk(
			"Exited (with tsp) _Z9vectorAddPKfS0_Pf x=%lu duration=%llu tsp=%luns\n",
			x, delta, *tsp);
	} else {
		bpf_printk("Exited (without tsp) _Z9vectorAddPKfS0_Pf x=%lu \n",
			   x);
	}

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
