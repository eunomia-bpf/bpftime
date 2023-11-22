#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

SEC("uprobe/benchmark/test:__benchmark_test_function3")
int BPF_UPROBE(__benchmark_test_function3, const char *a, int b, uint64_t c)
{
	return b + c;
}

SEC("uretprobe/benchmark/test:__benchmark_test_function2")
int BPF_URETPROBE(__benchmark_test_function2, int ret)
{
	return ret;
}

SEC("uprobe/benchmark/test:__benchmark_test_function1")
int BPF_UPROBE(__benchmark_test_function1_1, const char *a, int b, uint64_t c)
{
	return b + c;
}

SEC("uretprobe/benchmark/test:__benchmark_test_function1")
int BPF_URETPROBE(__benchmark_test_function_1_2, int ret)
{
	return ret;
}

char LICENSE[] SEC("license") = "GPL";
