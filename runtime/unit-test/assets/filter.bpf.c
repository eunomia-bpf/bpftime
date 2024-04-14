#include "vmlinux.h"
#include "bpf/bpf_tracing.h"
#include "bpf/bpf_helpers.h"
#include "ufunc.bpf.h"

#ifndef BPF_UPROBE
#define BPF_UPROBE BPF_KPROBE
#endif
#ifndef BPF_URETPROBE
#define BPF_URETPROBE BPF_KRETPROBE
#endif

enum filter_op {
	OP_SKIP,
	OP_RESUME,
};

static const int (*test_pass_param)(char *str, char c,
				    long long parm1) = (void *)40;

uint64_t BPF_UPROBE(my_function, char *str, char c, long long parm1)
{
	if (str == NULL) {
		bpf_set_retval(-22);
		return OP_SKIP;
	}
	test_pass_param(str, c, parm1);

	return OP_RESUME;
}
