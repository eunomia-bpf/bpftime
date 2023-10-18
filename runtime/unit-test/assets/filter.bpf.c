#include "vmlinux.h"
#include "bpf/bpf_tracing.h"
#include "bpf/bpf_helpers.h"
#include "ffi.bpf.h"

enum PatchOp {
	OP_SKIP,
	OP_RESUME,
};

static const int (*test_pass_param)(char *str, char c, long long parm1) = (void *)4097;

uint64_t BPF_UPROBE(my_function, char *str, char c, long long parm1)
{
	if (str == NULL) {
		return OP_SKIP;
	}
	test_pass_param(str, c, parm1);
	bpf_set_retval(-22);
	return OP_RESUME;
}
