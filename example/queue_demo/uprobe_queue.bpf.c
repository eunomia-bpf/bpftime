#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct event_data {
	u64 timestamp;
	u32 pid;
	u32 tid;
	u32 counter;
	u32 function_id; // 1=target_function, 2=secondary_function
	s32 input_value;
	char comm[16];
};

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

SEC("uprobe/./target:target_function")
int target_function_entry(struct pt_regs *ctx)
{
	struct event_data event = {};

	event.timestamp = bpf_ktime_get_ns();
	event.pid = bpf_get_current_pid_tgid() >> 32;
	event.tid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
	event.function_id = 1;
	bpf_get_current_comm(&event.comm, sizeof(event.comm));
	event.input_value = (s32)PT_REGS_PARM1(ctx);

	// Update call counter
	u32 key = 0;
	u32 *count = bpf_map_lookup_elem(&call_counter, &key);
	if (count) {
		*count += 1;
		event.counter = *count;
	} else {
		u32 init_count = 1;
		bpf_map_update_elem(&call_counter, &key, &init_count, BPF_ANY);
		event.counter = 1;
	}

	// Push event to queue
	long ret = bpf_map_push_elem(&events_queue, &event, BPF_ANY);
	if (ret != 0) {
		bpf_printk("Failed to push event to queue, ret=%ld\n", ret);
	} else {
		bpf_printk(
			"Pushed event to queue: pid=%d, counter=%d, input=%d\n",
			event.pid, event.counter, event.input_value);
	}

	return 0;
}

SEC("uprobe/./target:secondary_function")
int secondary_function_entry(struct pt_regs *ctx)
{
	struct event_data event = {};

	event.timestamp = bpf_ktime_get_ns();
	event.pid = bpf_get_current_pid_tgid() >> 32;
	event.tid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
	event.function_id = 2;
	bpf_get_current_comm(&event.comm, sizeof(event.comm));
	event.input_value = 0;

	// Update call counter
	u32 key = 0;
	u32 *count = bpf_map_lookup_elem(&call_counter, &key);
	if (count) {
		*count += 1;
		event.counter = *count;
	} else {
		u32 init_count = 1;
		bpf_map_update_elem(&call_counter, &key, &init_count, BPF_ANY);
		event.counter = 1;
	}

	// Push event to queue
	long ret = bpf_map_push_elem(&events_queue, &event, BPF_ANY);
	if (ret != 0) {
		bpf_printk("Failed to push secondary event to queue, ret=%ld\n",
			   ret);
	} else {
		bpf_printk(
			"Pushed secondary event to queue: pid=%d, counter=%d\n",
			event.pid, event.counter);
	}

	return 0;
}

char LICENSE[] SEC("license") = "GPL";