#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>


SEC("uretprobe/benchmark/test:__benchmark_test_function3")
int BPF_URETPROBE(__benchmark_test_function, const char *a, int b, uint64_t c)
{
	return b + c;
}

char LICENSE[] SEC("license") = "GPL";
