#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// Statistics index definitions (must match userspace)
#define STAT_TOTAL_ACCESSES 0
#define STAT_UNIQUE_USERS 1
#define STAT_REPEAT_USERS 2
#define STAT_ADMIN_OPS 3
#define STAT_SYSTEM_EVENTS 4
#define STAT_BLOOM_HITS 5
#define STAT_BLOOM_MISSES 6

// Bloom filter map for tracking user IDs
struct {
	__uint(type, BPF_MAP_TYPE_BLOOM_FILTER);
	__uint(max_entries, 1000);
	__uint(value_size, sizeof(u32));
	__uint(map_extra, 3); // Number of hash functions
} user_bloom_filter SEC(".maps");

// Statistics map
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 8);
	__type(key, u32);
	__type(value, u64);
} stats SEC(".maps");

// Helper function to increment statistics
static inline void increment_stat(u32 index)
{
	u64 *count = bpf_map_lookup_elem(&stats, &index);
	if (count) {
		(*count)++;
	} else {
		u64 init_count = 1;
		bpf_map_update_elem(&stats, &index, &init_count, BPF_ANY);
	}
}

SEC("uprobe/./target:user_access")
int user_access_entry(struct pt_regs *ctx)
{
	u32 user_id = (u32)PT_REGS_PARM1(ctx);

	// Increment total accesses
	increment_stat(STAT_TOTAL_ACCESSES);

	// Check if user ID exists in bloom filter
	void *exists = bpf_map_lookup_elem(&user_bloom_filter, &user_id);

	if (exists) {
		// Bloom filter hit - user might exist (possible false positive)
		increment_stat(STAT_REPEAT_USERS);
		increment_stat(STAT_BLOOM_HITS);
		bpf_printk("Bloom filter HIT for user_id=%d (repeat user)\n",
			   user_id);
	} else {
		// Bloom filter miss - user definitely doesn't exist (no false
		// negatives)
		increment_stat(STAT_UNIQUE_USERS);
		increment_stat(STAT_BLOOM_MISSES);

		// Add user to bloom filter
		u32 value = 1; // Dummy value for bloom filter
		long ret = bpf_map_update_elem(&user_bloom_filter, &user_id,
					       &value, BPF_ANY);
		if (ret != 0) {
			bpf_printk(
				"Failed to add user_id=%d to bloom filter, ret=%ld\n",
				user_id, ret);
		} else {
			bpf_printk(
				"Bloom filter MISS for user_id=%d (new user, added to filter)\n",
				user_id);
		}
	}

	return 0;
}

SEC("uprobe/./target:admin_operation")
int admin_operation_entry(struct pt_regs *ctx)
{
	increment_stat(STAT_ADMIN_OPS);

	u32 admin_id = (u32)PT_REGS_PARM1(ctx);
	bpf_printk("Admin operation by admin_id=%d\n", admin_id);

	return 0;
}

SEC("uprobe/./target:system_event")
int system_event_entry(struct pt_regs *ctx)
{
	increment_stat(STAT_SYSTEM_EVENTS);

	bpf_printk("System event triggered\n");

	return 0;
}

char LICENSE[] SEC("license") = "GPL";