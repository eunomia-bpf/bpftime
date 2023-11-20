#ifndef BPFTIME_UREPLACE_ATTACH_H
#define BPFTIME_UREPLACE_ATTACH_H

#include <unistd.h>
#include <stdlib.h>
#include <syscall.h>
#include <linux/perf_event.h>
#include <linux/bpf.h>
#include <bpf/bpf.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

long elf_find_func_offset_from_file(const char *binary_path, const char *name);

#define PERF_UPROBE_REF_CTR_OFFSET_BITS 32
#define PERF_UPROBE_REF_CTR_OFFSET_SHIFT 32
#define BPF_TYPE_UFILTER 8

static inline __u64 ptr_to_u64(const void *ptr)
{
	return (__u64)(unsigned long)ptr;
}

static int perf_event_open_ureplace(const char *name, uint64_t offset, int pid,
				    size_t ref_ctr_off)
{
	const size_t attr_sz = sizeof(struct perf_event_attr);
	struct perf_event_attr attr;
	int type, pfd;

	if ((__u64)ref_ctr_off >= (1ULL << PERF_UPROBE_REF_CTR_OFFSET_BITS))
		return -EINVAL;

	memset(&attr, 0, attr_sz);

	type = BPF_TYPE_UFILTER;
	attr.size = attr_sz;
	attr.type = type;
	attr.config |= (__u64)ref_ctr_off << PERF_UPROBE_REF_CTR_OFFSET_SHIFT;
	attr.config1 = ptr_to_u64(name); /* kprobe_func or uprobe_path */
	attr.config2 = offset; /* kprobe_addr or probe_offset */

	/* pid filter is meaningful only for uprobes */
	pfd = syscall(__NR_perf_event_open, &attr, pid < 0 ? -1 : pid /* pid */,
		      pid == -1 ? 0 : -1 /* cpu */, -1 /* group_fd */,
		      PERF_FLAG_FD_CLOEXEC);
	return pfd >= 0 ? pfd : -errno;
}

static int bpf_prog_attach_ureplace(int prog_fd, const char *binary_path,
				    const char *name)
{
	int offset = elf_find_func_offset_from_file("./victim", "target_func");
	if (offset < 0) {
		return offset;
	}
	printf("offset: %d", offset);
	int res = perf_event_open_ureplace("./victim", offset, -1, 0);
	if (res < 0) {
		printf("perf_event_open_ureplace failed: %d\n", res);
		return res;
	}
	res = bpf_prog_attach(prog_fd, res, BPF_MODIFY_RETURN, 0);
	if (res < 0) {
		printf("bpf_prog_attach failed: %d\n", res);
		return res;
	}
	return 0;
}

#endif // BPFTIME_UREPLACE_ATTACH_H
