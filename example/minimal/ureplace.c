// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/* Copyright (c) 2020 Facebook */
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <stdlib.h>
#include "ureplace.skel.h"
#include <inttypes.h>
#include <syscall.h>
#include <linux/perf_event.h>

#define warn(...) fprintf(stderr, __VA_ARGS__)

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

static void sig_handler(int sig)
{
	exiting = true;
}

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

int main(int argc, char **argv)
{
	struct ureplace_bpf *skel;
	int err;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = ureplace_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = ureplace_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}
	err = bpf_prog_attach_ureplace(bpf_program__fd(skel->progs.do_ureplace_patch),
				       "./victim", "target_func");
	if (err) {
		fprintf(stderr, "Failed to attach BPF program\n");
		goto cleanup;
	}
	while (!exiting) {
		sleep(1);
	}
cleanup:
	/* Clean up */
	ureplace_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
