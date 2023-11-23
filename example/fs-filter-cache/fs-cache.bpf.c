/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include "fs-cache.h"

#define EXTENDED_HELPER_GET_ABS_PATH_ID 1003
#define EXTENDED_HELPER_PATH_JOIN_ID 1004

static void *(*bpftime_get_abs_path)(const char *filename, const char *buffer,
				     u64 size) = (void *)
	EXTENDED_HELPER_GET_ABS_PATH_ID;

static void *(*bpftime_path_join)(const char *filename1, const char *filename2,
				  const char *buffer, u64 size) = (void *)
	EXTENDED_HELPER_PATH_JOIN_ID;

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
	char fname[NAME_MAX];
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
	bpf_probe_read_str(args.fname, sizeof(args.fname),
			   (const void *)ctx->args[0]);
	args.flags = (int)ctx->args[1];
	bpf_map_update_elem(&open_args_start, &id, &args, 0);
	// bpf_printk("sys_enter_open %s\n", args.fname);
	return 0;
}

#define AT_FDCWD -100

SEC("tracepoint/syscalls/sys_enter_openat")
int tracepoint__syscalls__sys_enter_openat(struct trace_event_raw_sys_enter *ctx)
{
	u64 id = bpf_get_current_pid_tgid();

	/* store arg info for later lookup */
	struct open_args_t args = {};
	int fd = (int)ctx->args[0];
	bpf_probe_read_str(args.fname, sizeof(args.fname),
			   (const void *)ctx->args[1]);
	args.flags = (int)ctx->args[2];
	if (fd != AT_FDCWD) {
		struct dir_fd_data *dir_data = get_dir_fd_data(fd);
		if (dir_data == NULL) {
			return 0;
		}
		bpftime_path_join(dir_data->dir_path, args.fname, args.fname,
				  sizeof(args.fname));
	}
	bpf_map_update_elem(&open_args_start, &id, &args, 0);
	// bpf_printk("sys_enter_openat %d %s\n", fd, args.fname);
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
	bpftime_get_abs_path(ap->fname, data.dir_path, sizeof(data.dir_path));
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

struct getdents64_buffer empty_buf = {};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, struct dir_fd_data);
	__type(value, struct getdents64_buffer);
} getdents64_cache_map SEC(".maps");

struct getdents_args_t {
	int fd;
	void *dirp;
	unsigned int count;
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u64);
	__type(value, struct getdents_args_t);
} getdents64_args_start SEC(".maps");

SEC("tracepoint/syscalls/sys_enter_getdents64")
int tracepoint__syscalls__sys_enter_getdents64(
	struct trace_event_raw_sys_enter *ctx)
{
	int fd = (int)ctx->args[0];
	// check the fd is a dir fd
	struct dir_fd_data *dir_data = get_dir_fd_data(fd);
	if (dir_data == NULL) {
		return 0;
	}
	// bpf_printk("trace_enter sys_enter_getdents64 fd: %d, path %s\n", fd,
	// 	   dir_data->dir_path);
	struct getdents_args_t args = {};
	args.fd = fd;
	args.dirp = (void *)ctx->args[1];
	args.count = (unsigned int)ctx->args[2];
	struct getdents64_buffer *buf =
		bpf_map_lookup_elem(&getdents64_cache_map, dir_data);
	u64 id = bpf_get_current_pid_tgid();
	if (buf) {
		if (id == buf->last_pid_tgid) {
			// skip if the lookup is duplicated
			// bpf_printk(
			// 	"trace_enter sys_enter_getdents64 cache skip fd: %d, path %s\n", fd, dir_data->dir_path);
			bpf_override_return((void *)ctx, 0);
			buf->last_pid_tgid = 0;
			return 0;
		}
		// cache exists
		// bpf_printk(
		// 	"trace_enter sys_enter_getdents64 cache exists fd: %d, path %s\n",
		// 	fd, dir_data->dir_path);
		bpf_probe_write_user((void *)args.dirp, buf->buf,
				     DENTS_BUF_SIZE > args.count ?
					     args.count :
					     DENTS_BUF_SIZE);
		bpf_override_return((void *)ctx, buf->nread);
		buf->last_pid_tgid = id;
		return 0;
	}
	bpf_map_update_elem(&getdents64_args_start, &id, &args, 0);
	return 0;
}

SEC("tracepoint/syscalls/sys_exit_getdents64")
int tracepoint__syscalls__sys_exit_getdents64(
	struct trace_event_raw_sys_exit *ctx)
{
	int ret = ctx->ret;
	if (ret < 0) {
		return 0;
	}
	u64 id = bpf_get_current_pid_tgid();
	struct getdents_args_t *args =
		bpf_map_lookup_elem(&getdents64_args_start, &id);
	if (!args) {
		return 0;
	}
	// check the fd is a dir fd
	struct dir_fd_data *dir_data = get_dir_fd_data(args->fd);
	if (dir_data == NULL) {
		return 0;
	}
	// bpf_printk("sys_exit_getdents64 fd: %d, path %s\n", args->fd,
	// 	   dir_data->dir_path);
	struct getdents64_buffer *buf =
		bpf_map_lookup_elem(&getdents64_cache_map, dir_data);
	if (buf) {
		// cache exists
		return 0;
	}
	// bpf_printk("sys_exit_getdents64 cache not exists fd: %d, path %s,
	// count %d\n", 	   args->fd, dir_data->dir_path, args->count);
	// if (DENTS_BUF_SIZE <= args->count) {
	// 	// too big, don't cache
	// 	return 0;
	// }
	bpf_map_update_elem(&getdents64_cache_map, dir_data, &empty_buf, 0);
	buf = bpf_map_lookup_elem(&getdents64_cache_map, dir_data);
	if (!buf) {
		return 0;
	}
	bpf_probe_read(buf->buf, DENTS_BUF_SIZE, args->dirp);
	buf->nread = ret;
	// int nread = ret;
	// struct linux_dirent64 *d;
	// char *dirent64_buf = buf->buf;
	// for (int bpos = 0; bpos < nread;) {
	// 	d = (struct linux_dirent64 *)(dirent64_buf + bpos);
	// 	bpf_printk("inode=%ld name=%s\n",
	// 	       (long)d->d_ino,
	// 	       d->d_name);
	// 	bpos += d->d_reclen;
	// }
	buf->last_pid_tgid = id;
	// bpf_printk("sys_exit_getdents64 successful update fd: %d, path %s\n",
	// 	   args->fd, dir_data->dir_path);
	return 0;
}

SEC("tracepoint/syscalls/sys_enter_newfstatat")
int tracepoint__syscalls__sys_enter_newfstatat(
	struct trace_event_raw_sys_enter *ctx)
{
	int fd = (int)ctx->args[0];
	struct dir_fd_data *dir_data = get_dir_fd_data(fd);
	if (dir_data == NULL) {
		return 0;
	}
	const char *path = (const char *)ctx->args[1];
	char buffer[NAME_MAX] = {};
	bpf_probe_read_str(buffer, sizeof(buffer), path);
	// bpf_printk("sys_enter_newfstatat fd: %d, path %s, dir path %s\n", fd,
	// 	   buffer, dir_data->dir_path);
	return 0;
}

SEC("tracepoint/syscalls/sys_enter_statfs")
int tracepoint__syscalls__sys_enter_statfs(struct trace_event_raw_sys_enter *ctx)
{
	// bpf_printk("trace_enter sys_enter_statfs\n");
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
