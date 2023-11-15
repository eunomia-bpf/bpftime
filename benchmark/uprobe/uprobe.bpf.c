#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

SEC("uprobe/benchmark/test:__benchmark_test_function2")
int BPF_PROG(bpf_benchmark_test_function)
{
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
