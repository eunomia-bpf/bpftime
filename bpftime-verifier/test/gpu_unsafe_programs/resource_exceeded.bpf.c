/*
 * Safety property: Violates the per-hook helper-call budget. This program
 * contains more than 64 helper invocations in a normal kprobe-style GPU hook.
 * Expected verifier result: REJECT.
 * Why this matters for GPU execution: even memory-safe helper-heavy hooks can
 * consume enough device resources to perturb kernel execution and create
 * denial-of-service style slowdown.
 */

#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} helper_budget_sink SEC(".maps");

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;

#define SAMPLE_STEP(n) total += (bpf_get_globaltimer() ^ (n))

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__resource_exceeded()
{
	u32 key = 0;
	u64 total = 0;

	/*
	 * Intentionally spelled out to exceed the static helper-call budget.
	 * 70 globaltimer helpers + 1 map_update helper => >64 helpers total.
	 */
	SAMPLE_STEP(0);
	SAMPLE_STEP(1);
	SAMPLE_STEP(2);
	SAMPLE_STEP(3);
	SAMPLE_STEP(4);
	SAMPLE_STEP(5);
	SAMPLE_STEP(6);
	SAMPLE_STEP(7);
	SAMPLE_STEP(8);
	SAMPLE_STEP(9);
	SAMPLE_STEP(10);
	SAMPLE_STEP(11);
	SAMPLE_STEP(12);
	SAMPLE_STEP(13);
	SAMPLE_STEP(14);
	SAMPLE_STEP(15);
	SAMPLE_STEP(16);
	SAMPLE_STEP(17);
	SAMPLE_STEP(18);
	SAMPLE_STEP(19);
	SAMPLE_STEP(20);
	SAMPLE_STEP(21);
	SAMPLE_STEP(22);
	SAMPLE_STEP(23);
	SAMPLE_STEP(24);
	SAMPLE_STEP(25);
	SAMPLE_STEP(26);
	SAMPLE_STEP(27);
	SAMPLE_STEP(28);
	SAMPLE_STEP(29);
	SAMPLE_STEP(30);
	SAMPLE_STEP(31);
	SAMPLE_STEP(32);
	SAMPLE_STEP(33);
	SAMPLE_STEP(34);
	SAMPLE_STEP(35);
	SAMPLE_STEP(36);
	SAMPLE_STEP(37);
	SAMPLE_STEP(38);
	SAMPLE_STEP(39);
	SAMPLE_STEP(40);
	SAMPLE_STEP(41);
	SAMPLE_STEP(42);
	SAMPLE_STEP(43);
	SAMPLE_STEP(44);
	SAMPLE_STEP(45);
	SAMPLE_STEP(46);
	SAMPLE_STEP(47);
	SAMPLE_STEP(48);
	SAMPLE_STEP(49);
	SAMPLE_STEP(50);
	SAMPLE_STEP(51);
	SAMPLE_STEP(52);
	SAMPLE_STEP(53);
	SAMPLE_STEP(54);
	SAMPLE_STEP(55);
	SAMPLE_STEP(56);
	SAMPLE_STEP(57);
	SAMPLE_STEP(58);
	SAMPLE_STEP(59);
	SAMPLE_STEP(60);
	SAMPLE_STEP(61);
	SAMPLE_STEP(62);
	SAMPLE_STEP(63);
	SAMPLE_STEP(64);
	SAMPLE_STEP(65);
	SAMPLE_STEP(66);
	SAMPLE_STEP(67);
	SAMPLE_STEP(68);
	SAMPLE_STEP(69);

	bpf_map_update_elem(&helper_budget_sink, &key, &total, BPF_ANY);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
