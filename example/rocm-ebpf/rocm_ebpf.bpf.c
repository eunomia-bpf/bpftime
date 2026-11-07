// SPDX-License-Identifier: MIT
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include "bpf_test_println.h"

SEC("kprobe/rocm__bpftime_example_kernel")
int entry__rocm(void *ctx)
{
	char message[] = "BPF entry";
	bpf_test_println(message);
	return 0;
}

SEC("kretprobe/rocm__bpftime_example_kernel")
int exit__rocm(void *ctx)
{
	char message[] = "BPF exit";
	bpf_test_println(message);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
