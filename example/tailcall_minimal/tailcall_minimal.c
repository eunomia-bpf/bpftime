// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/* Copyright (c) 2020 Facebook */
#include "linux/bpf.h"
#include "linux/filter.h"
#include "bpf/bpf.h"
#include <asm/unistd_64.h>
#include <errno.h>
#include <string.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <stdlib.h>
#include "./.output/tailcall_minimal.skel.h"
#include <inttypes.h>
#define warn(...) fprintf(stderr, __VA_ARGS__)

#define BPF_LD_IMM64_RAW_FULL(DST, SRC, OFF1, OFF2, IMM1, IMM2)                \
	((struct bpf_insn){ .code = BPF_LD | BPF_DW | BPF_IMM,                 \
			    .dst_reg = DST,                                    \
			    .src_reg = SRC,                                    \
			    .off = OFF1,                                       \
			    .imm = IMM1 }),                                    \
		((struct bpf_insn){ .code = 0,                                 \
				    .dst_reg = 0,                              \
				    .src_reg = 0,                              \
				    .off = OFF2,                               \
				    .imm = IMM2 })
#define BPF_STX_MEM(SIZE, DST, SRC, OFF)                                       \
	((struct bpf_insn){ .code = BPF_STX | BPF_SIZE(SIZE) | BPF_MEM,        \
			    .dst_reg = DST,                                    \
			    .src_reg = SRC,                                    \
			    .off = OFF,                                        \
			    .imm = 0 })

#define BPF_ALU64_IMM(OP, DST, IMM)                                            \
	((struct bpf_insn){ .code = BPF_ALU64 | BPF_OP(OP) | BPF_K,            \
			    .dst_reg = DST,                                    \
			    .src_reg = 0,                                      \
			    .off = 0,                                          \
			    .imm = IMM })

#define BPF_EXIT_INSN()                                                        \
	((struct bpf_insn){ .code = BPF_JMP | BPF_EXIT,                        \
			    .dst_reg = 0,                                      \
			    .src_reg = 0,                                      \
			    .off = 0,                                          \
			    .imm = 0 })

#define BPF_EMIT_CALL(FUNC)                                                    \
	((struct bpf_insn){ .code = BPF_JMP | BPF_CALL,                        \
			    .dst_reg = 0,                                      \
			    .src_reg = 0,                                      \
			    .off = 0,                                          \
			    .imm = ((FUNC)-BPF_FUNC_unspec) })

#define BPF_MOV64_REG(DST, SRC)                                                \
	((struct bpf_insn){ .code = BPF_ALU64 | BPF_MOV | BPF_X,               \
			    .dst_reg = DST,                                    \
			    .src_reg = SRC,                                    \
			    .off = 0,                                          \
			    .imm = 0 })
#define BPF_MOV64_IMM(DST, IMM)                                                \
	((struct bpf_insn){ .code = BPF_ALU64 | BPF_MOV | BPF_K,               \
			    .dst_reg = DST,                                    \
			    .src_reg = 0,                                      \
			    .off = 0,                                          \
			    .imm = IMM })
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

static long my_bpf_syscall(long cmd, union bpf_attr *attr, unsigned long size)
{
	int attempts = 5;
	long fd;
	do {
		__asm__ volatile("movq %1, %%rax\n"
				 "movq %2, %%rdi\n"
				 "movq %3, %%rsi\n"
				 "movq %4, %%rdx\n"
				 "movq $0, %%r10\n"
				 "movq $0, %%r8\n"
				 "movq $0, %%r9\n"
				 "syscall\n"
				 : "=a"(fd)
				 : "i"((long)__NR_bpf), "m"(cmd), "m"(attr),
				   "m"(size)
				 : "memory");
	} while (fd < 0 && fd == -EAGAIN && --attempts > 0);
	return fd;
}

static int load_program_into_kernel()
{
	puts("Loading program into kernel..");
	fflush(stdout);

	char log_buf[1 << 10];
	struct bpf_insn insns[] = {
		// r1 = 0x2164656b6f766e49 ll
		BPF_LD_IMM64_RAW_FULL(1, 0, 0, 0, 0x6f766e49, 0x2164656b),
		// *(u64 *)(r10 - 0x10) = r1
		BPF_STX_MEM(BPF_DW, 10, 1, -0x10),
		// r1 = 0x0
		BPF_MOV64_IMM(1, 0),
		// *(u8 *)(r10 - 0x8) = r1
		BPF_STX_MEM(BPF_B, 10, 1, -8),
		// r1 = r10
		BPF_MOV64_REG(1, 10),
		// r1 += -0x10
		BPF_ALU64_IMM(BPF_ADD, 1, -0x10),
		// r2 = 0x9
		BPF_MOV64_IMM(2, 9),
		// call 0x06
		BPF_EMIT_CALL(0x06),
		// r0 = 0
		BPF_MOV64_IMM(0, 0),
		// exit
		BPF_EXIT_INSN()
	};
	LIBBPF_OPTS(bpf_prog_load_opts, load_opts, .log_level = 1,
		    .log_size = sizeof(log_buf), .log_buf = log_buf);
	enum bpf_prog_type prog_ty;
	if (getenv("BPFTIME_USED")) {
		puts("Using prog type BPF_PROG_TYPE_RAW_TRACEPOINT");
		prog_ty = BPF_PROG_TYPE_RAW_TRACEPOINT;
	} else {
		puts("Using prog type BPF_PROG_TYPE_KPROBE");
		prog_ty = BPF_PROG_TYPE_KPROBE;
	}
	// We need to directly use the syscall, to avoid being hooked by bpftime
	// syscall server
	union bpf_attr attr;
	memset(&attr, 0, sizeof(attr));
	attr.prog_type = (int)prog_ty;
	attr.insn_cnt = sizeof(insns) / sizeof(insns[0]);
	attr.license = (uintptr_t) "GPL";
	attr.log_buf = (uintptr_t)log_buf;
	attr.log_level = 1;
	attr.log_size = sizeof(log_buf);
	attr.insns = (uintptr_t)insns;
	strcpy(attr.prog_name, "test_prog");
#ifndef offsetofend
#define offsetofend(TYPE, FIELD)                                               \
	(offsetof(TYPE, FIELD) + sizeof(((TYPE *)0)->FIELD))
#endif
	const size_t attr_sz = offsetofend(union bpf_attr, fd_array);
	int ret = my_bpf_syscall(BPF_PROG_LOAD, &attr, attr_sz);
	if (ret < 0) {
		fprintf(stderr, "Unable to load program into kernel: %d\n",
			ret);
		fprintf(stderr, "%s", log_buf);
	}
	puts("Loaded program into kernel..");
	fflush(stdout);
	return ret;
}

int main(int argc, char **argv)
{
	struct tailcall_minimal_bpf *skel;
	int err;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = tailcall_minimal_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}
	puts("Loading skeleton..");
	fflush(stdout);

	/* Load & verify BPF programs */
	err = tailcall_minimal_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}
	puts("Loaded skeleton..");
	fflush(stdout);
	int prog_fd = load_program_into_kernel();
	if (prog_fd < 0) {
		goto cleanup;
	}
	printf("Kernel program fd: %d\n", prog_fd);
	fflush(stdout);
	int idx = 0;
	int map_fd = bpf_map__fd(skel->maps.prog_array);
	err = bpf_map_update_elem(map_fd, &idx, &prog_fd, 0);
	if (err) {
		fprintf(stderr, "Unable to update prog array map: %d\n", errno);
		goto cleanup;
	}
	puts("Map updated");
	fflush(stdout);
	err = tailcall_minimal_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}
	puts("Attached");
	fflush(stdout);
	unsigned i = 0;
	while (!exiting) {
		sleep(1);
		i++;
		printf("See /sys/kernel/debug/tracing/trace_pipe for output (%u)\n",
		       i);
		fflush(stdout);
	}
cleanup:
	/* Clean up */
	tailcall_minimal_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
