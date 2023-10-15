// SPDX-License-Identifier: GPL-2.0
// Copyright (c) 2019 Facebook
// Copyright (c) 2020 Netflix
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
SEC("uretprobe/readline")
int BPF_URETPROBE(simple_probe, long ret)
{
	bpf_printk("Ret=%ld", ret);

	return 0;
}
char LICENSE[] SEC("license") = "GPL";
