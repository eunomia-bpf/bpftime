/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include "fs-filter-cache.h"

#define PID_MASK_FOR_PFD 0xffffffff00000000
#define FD_MASK_FOR_PFD 0x00000000ffffffff
#define MAKE_PFD(pid, fd)                                                      \
	(((unsigned long long)pid << 32) | (unsigned long long)fd)

struct dir_fd_data {
	char dir_path[NAME_MAX];
};

// pid & fd for all open dir related fds
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u64);
	__type(value, struct dir_fd_data);
} dir_fd_map SEC(".maps");

static __always_inline void set_dir_fd_data(u32 fd, struct dir_fd_data *data)
{
	if (fd < 0) {
		return;
	}
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 key = MAKE_PFD(pid, fd);
	// bpf_printk("set_dir_fd_data: %d %d, %s\n", pid, fd, data->dir_path);
	bpf_map_update_elem(&dir_fd_map, &key, data, 0);
}

static __always_inline struct dir_fd_data *get_dir_fd_data(u32 fd)
{
	if (fd < 0) {
		return NULL;
	}
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 key = MAKE_PFD(pid, fd);
	return bpf_map_lookup_elem(&dir_fd_map, &key);
}

static __always_inline void clear_dir_data_fd(int fd)
{
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 key = MAKE_PFD(pid, fd);
	bpf_map_delete_elem(&dir_fd_map, &key);
}

struct open_args_t {
	const char *fname;
	int flags;
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u64);
	__type(value, struct open_args_t);
} open_args_start SEC(".maps");

SEC("tracepoint/syscalls/sys_enter_open")
int tracepoint__syscalls__sys_enter_open(struct trace_event_raw_sys_enter *ctx)
{
	u64 id = bpf_get_current_pid_tgid();

	/* store arg info for later lookup */
	struct open_args_t args = {};
	args.fname = (const char *)ctx->args[0];
	args.flags = (int)ctx->args[1];
	bpf_map_update_elem(&open_args_start, &id, &args, 0);
	// bpf_printk("trace_enter sys_enter_open\n");
	return 0;
}

SEC("tracepoint/syscalls/sys_enter_openat")
int tracepoint__syscalls__sys_enter_openat(struct trace_event_raw_sys_enter *ctx)
{
	u64 id = bpf_get_current_pid_tgid();

	/* store arg info for later lookup */
	struct open_args_t args = {};
	args.fname = (const char *)ctx->args[1];
	args.flags = (int)ctx->args[2];
	bpf_map_update_elem(&open_args_start, &id, &args, 0);
	// bpf_printk("trace_enter sys_enter_openat\n");
	return 0;
}

static __always_inline int trace_exit(struct trace_event_raw_sys_exit *ctx)
{
	// bpf_printk("trace_exit open\n");
	int ret;
	u64 id = bpf_get_current_pid_tgid();

	struct open_args_t *ap = bpf_map_lookup_elem(&open_args_start, &id);
	if (!ap)
		return 0; /* missed entry */
	ret = ctx->ret;
	if (ret < 0) {
		// bpf_printk("trace_exit open: %d\n", ret);
		return 0;
	}
	struct dir_fd_data data = {};
	bpf_probe_read_str(&data.dir_path, sizeof(data.dir_path), ap->fname);
	set_dir_fd_data(ret, &data);
	return 0;
}

SEC("tracepoint/syscalls/sys_exit_open")
int tracepoint__syscalls__sys_exit_open(struct trace_event_raw_sys_exit *ctx)
{
	return trace_exit(ctx);
}

SEC("tracepoint/syscalls/sys_exit_openat")
int tracepoint__syscalls__sys_exit_openat(struct trace_event_raw_sys_exit *ctx)
{
	return trace_exit(ctx);
}

SEC("tracepoint/syscalls/sys_enter_close")
int tracepoint__syscalls__sys_enter_close(struct trace_event_raw_sys_enter *ctx)
{
	int fd = (int)ctx->args[0];
	// bpf_printk("trace_enter sys_enter_close: %d\n", fd);
	clear_dir_data_fd(fd);
	return 0;
}

SEC("tracepoint/syscalls/sys_enter_getdents64")
int tracepoint__syscalls__sys_enter_getdents64(
	struct trace_event_raw_sys_enter *ctx)
{
	int fd = (int)ctx->args[0];
	struct dir_fd_data *dir_data = get_dir_fd_data(fd);
	if (dir_data == NULL) {
		return 0;
	}
	bpf_printk("trace_enter sys_enter_getdents64 fd: %d, path %s\n", fd,
		   dir_data->dir_path);
	return 0;
}

SEC("tracepoint/syscalls/sys_enter_newfstatat")
int tracepoint__syscalls__sys_enter_newfstatat(
	struct trace_event_raw_sys_enter *ctx)
{
	// bpf_printk("trace_enter sys_enter_newfstatat\n");
	return 0;
}

SEC("tracepoint/syscalls/sys_enter_statfs")
int tracepoint__syscalls__sys_enter_statfs(struct trace_event_raw_sys_enter *ctx)
{
	// bpf_printk("trace_enter sys_enter_statfs\n");
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
