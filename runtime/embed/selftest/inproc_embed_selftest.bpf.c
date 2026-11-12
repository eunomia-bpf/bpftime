// SPDX-License-Identifier: Apache-2.0
// Map-less uprobe program used by the single-process embed selftest. It echoes
// the low 16 bits of the first argument (PT_REGS_PARM1) back through R0, so the
// harness can prove trigger -> synthesized pt_regs -> BPF execution without any
// ringbuf/PID/map relocation machinery.
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char LICENSE[] SEC("license") = "GPL";

SEC("uprobe")
int inproc_echo_parm1(struct pt_regs *ctx)
{
	return PT_REGS_PARM1(ctx) & 0xffff;
}
