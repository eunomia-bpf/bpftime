// SPDX-License-Identifier: GPL-2.0
// Copyright (c) 2019 Facebook
// Copyright (c) 2020 Netflix
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
SEC("uretprobe/./example/simple_uretprobe_test/victim:simple_add")
int BPF_URETPROBE(simple_probe, long ret)
{
	bpf_printk("Ret=%ld\n", ret);

	return 0;
}
char LICENSE[] SEC("license") = "GPL";
