/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef __SYSCALL_TRACER_H
#define __SYSCALL_TRACER_H

#define TASK_COMM_LEN 16
#define NAME_MAX 255
#define INVALID_UID ((uid_t)-1)
#define MAX_INSN_SIZE 128
#define BPF_OBJ_NAME_LEN 16U
#define MAX_FILENAME_LEN 127


enum event_type {
	SYS_OPEN,
	SYS_CLOSE,
	SYS_BPF,
	SYS_IOCTL,
	SYS_PERF_EVENT_OPEN,
	BPF_PROG_LOAD_EVENT,
	EXEC_EXIT,
};

enum bpf_fd_type {
	BPF_FD_TYPE_OTHERS,
	BPF_FD_TYPE_PERF,
	BPF_FD_TYPE_PROG,
	BPF_FD_TYPE_MAP,
	BPF_FD_TYPE_MAX,
};

struct bpf_fd_data {
	enum bpf_fd_type type;
	// porg id or map id;
	int kernel_id;
};

struct event {
	enum event_type type;

	// basic info
	int pid;
	char comm[TASK_COMM_LEN];

	union {
		struct {
			int ret;
			unsigned int bpf_cmd;
			union bpf_attr attr;
			unsigned int attr_size;
			// additional field for getting map id
			int map_id;
		} bpf_data;

		struct {
			int ret;
			int flags;
			char fname[NAME_MAX];
		} open_data;

		struct {
			int ret;
			int pid;
			int cpu;

			// uprobe data
			char name_or_path[NAME_MAX];
			struct perf_event_attr attr;
		} perf_event_data;

		struct {
			unsigned int type;
			unsigned int insn_cnt;
			char prog_name[BPF_OBJ_NAME_LEN];
			// used as key for later lookup in userspace
			unsigned long long insns_ptr;
		} bpf_loaded_prog;

		struct {
			int fd;
			struct bpf_fd_data fd_data;
		} close_data;

		struct {
			int fd;
			unsigned long req;
			int data;
			int ret;

			int bpf_prog_id;
		} ioctl_data;

		struct {
			int exit_event;
			int ppid;
			unsigned exit_code;
			unsigned long long time_ns;
			char filename[MAX_FILENAME_LEN];
		} exec_data;
	};
};

#define BPF_COMPLEXITY_LIMIT_INSNS 4096

struct bpf_insn_data {
	unsigned char code[BPF_COMPLEXITY_LIMIT_INSNS * sizeof(struct bpf_insn)];
	unsigned int code_len;
};

#define PID_MASK_FOR_PFD 0xffffffff00000000
#define FD_MASK_FOR_PFD 0x00000000ffffffff
#define MAKE_PFD(pid, fd) (((unsigned long long)pid << 32) | (unsigned long long)fd)

#endif /* __SYSCALL_TRACER_H */
