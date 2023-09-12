#define BPF_NO_GLOBAL_DATA
#include "vmlinux.h"
#include "bpf/bpf_tracing.h"
#include "bpf/bpf_helpers.h"
#include "ffi.bpf.h"

enum hotpatch_op {
	OP_SKIP,
	OP_RESUMUE,
};

SEC("uprobe/vim:openscript")
int BPF_UPROBE(openscript, char *name, int directly)
{
	bpf_printk("openscript: %s %d\n", name, directly);

	// Disallow sourcing a file in the sandbox, the commands would be
	// executed later, possibly outside of the sandbox.
	int res = FFI_CALL_NAME_0("check_secure");
	if (!res) {
		bpf_printk("check_secure return %d", res);
		return OP_SKIP;
	}
	return OP_RESUMUE;
}
