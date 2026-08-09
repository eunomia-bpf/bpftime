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
int cuda__probe()
{
	// This probe runs once per GPU thread. To keep the demo usable for
	// real-world grids (e.g. 4096x256 threads), only sample a single
	// "leader" thread per kernel launch: block(0,0,0), thread(0,0,0).
	u64 bx, by, bz;
	u64 tx, ty, tz;
	bpf_get_block_idx(&bx, &by, &bz);
	bpf_get_thread_idx(&tx, &ty, &tz);
	if (bx != 0 || by != 0 || bz != 0 || tx != 0 || ty != 0 || tz != 0)
		return 0;

	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 ts = bpf_get_globaltimer();

	// Store entry timestamp, keyed by pid (one record per process)
	bpf_map_update_elem(&start_ts, &pid, &ts, BPF_ANY);

	// Increment call count
	u64 *cnt = bpf_map_lookup_elem(&call_count, &pid);
	u64 new_cnt = cnt ? (*cnt + 1) : 1;
	bpf_map_update_elem(&call_count, &pid, &new_cnt, BPF_ANY);

	return 0;
}

SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int cuda__retprobe()
{
	// Same leader-thread filter as entry probe.
	u64 bx, by, bz;
	u64 tx, ty, tz;
	bpf_get_block_idx(&bx, &by, &bz);
	bpf_get_thread_idx(&tx, &ty, &tz);
	if (bx != 0 || by != 0 || bz != 0 || tx != 0 || ty != 0 || tz != 0)
		return 0;

	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 *tsp = bpf_map_lookup_elem(&start_ts, &pid);
	if (tsp) {
		u64 delta = bpf_get_globaltimer() - *tsp;
		bpf_map_delete_elem(&start_ts, &pid);
		u64 *total = bpf_map_lookup_elem(&total_time_ns, &pid);
		u64 new_total = total ? (*total + delta) : delta;
		bpf_map_update_elem(&total_time_ns, &pid, &new_total, BPF_ANY);
	}

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
