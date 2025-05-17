#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u32);
} test_hash_map SEC(".maps");

static const void (*ebpf_puts)(const char *) = 501;

SEC("kretprobe/_Z43matrix_multiply_add_bias_relu_kernel_SERIALPKfS0_S0_Pfiiib")
int retprobe__cuda(const char *call_str)
{
	bpf_printk(
		"Exiting _Z43matrix_multiply_add_bias_relu_kernel_SERIALPKfS0_S0_Pfiiib\n");
	u32 key = 1234;
	while (bpf_map_lookup_elem(&test_hash_map, &key) == NULL) {
	}
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
