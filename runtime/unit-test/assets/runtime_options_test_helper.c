#include <linux/bpf.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>

int main(void)
{
	const char license[] = "GPL";
	struct bpf_insn insns[1] = {};
	union bpf_attr attr = {
		.prog_type = BPF_PROG_TYPE_SOCKET_FILTER,
		.insn_cnt = 1,
		.insns = (uint64_t)(uintptr_t)insns,
		.license = (uint64_t)(uintptr_t)license,
		.prog_name = "bypass_me",
	};

	int fd = syscall(__NR_bpf, BPF_PROG_LOAD, &attr, sizeof(attr));
	if (fd >= 0)
		close(fd);
	return fd < 0;
}
