/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 *
 * Drives the intercepted bpf() syscall (LD_PRELOAD of bpftime-syscall-server)
 * to observe the errno and key handling of BPF_MAP_LOOKUP_ELEM and
 * BPF_MAP_LOOKUP_AND_DELETE_ELEM from a real caller. Each check prints one
 * "name=value" line; the Catch2 test parses them.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/bpf.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

static int bpf_call(int cmd, union bpf_attr *attr)
{
	return (int)syscall(__NR_bpf, cmd, attr, sizeof(*attr));
}

static int map_create(int type, unsigned int key_size,
		      unsigned int value_size, unsigned int max_entries,
		      unsigned int flags)
{
	union bpf_attr attr;
	memset(&attr, 0, sizeof(attr));
	attr.map_type = type;
	attr.key_size = key_size;
	attr.value_size = value_size;
	attr.max_entries = max_entries;
	attr.map_flags = flags;
	return bpf_call(BPF_MAP_CREATE, &attr);
}

static int map_update(int fd, const void *key, const void *value,
		      unsigned long long flags)
{
	union bpf_attr attr;
	memset(&attr, 0, sizeof(attr));
	attr.map_fd = fd;
	attr.key = (unsigned long long)(unsigned long)key;
	attr.value = (unsigned long long)(unsigned long)value;
	attr.flags = flags;
	return bpf_call(BPF_MAP_UPDATE_ELEM, &attr);
}

static int map_lookup(int fd, const void *key, void *value)
{
	union bpf_attr attr;
	memset(&attr, 0, sizeof(attr));
	attr.map_fd = fd;
	attr.key = (unsigned long long)(unsigned long)key;
	attr.value = (unsigned long long)(unsigned long)value;
	return bpf_call(BPF_MAP_LOOKUP_ELEM, &attr);
}

static int map_lookup_and_delete(int fd, const void *key, void *value)
{
	union bpf_attr attr;
	memset(&attr, 0, sizeof(attr));
	attr.map_fd = fd;
	attr.key = (unsigned long long)(unsigned long)key;
	attr.value = (unsigned long long)(unsigned long)value;
	return bpf_call(BPF_MAP_LOOKUP_AND_DELETE_ELEM, &attr);
}

/* A write-only map must reject an fd-based read with EPERM, not ENOENT: the
 * backend knows why the lookup failed and the syscall layer must not
 * overwrite that. */
static int check_lookup_errno(void)
{
	int fd = map_create(BPF_MAP_TYPE_LRU_HASH, sizeof(unsigned int),
			    sizeof(unsigned long long), 16, BPF_F_WRONLY);
	if (fd < 0) {
		printf("create_ok=0 create_errno=%d\n", errno);
		return 0;
	}
	unsigned int key = 7;
	unsigned long long value = 0x1122334455667788ULL;
	if (map_update(fd, &key, &value, BPF_ANY) != 0) {
		printf("update_ok=0 update_errno=%d\n", errno);
		return 0;
	}
	unsigned long long out = 0;
	errno = 0;
	int ret = map_lookup(fd, &key, &out);
	printf("lookup_ret=%d\n", ret);
	printf("lookup_errno=%d\n", errno);

	/* An absent key on an unrestricted map still reports ENOENT, so the
	 * EPERM above is the backend's own verdict and not a blanket one. */
	int normal_fd = map_create(BPF_MAP_TYPE_LRU_HASH, sizeof(unsigned int),
				   sizeof(unsigned long long), 16, 0);
	if (normal_fd < 0) {
		printf("normal_create_ok=0 normal_create_errno=%d\n", errno);
		return 0;
	}
	unsigned int absent_key = 99;
	errno = 0;
	ret = map_lookup(normal_fd, &absent_key, &out);
	printf("missing_ret=%d\n", ret);
	printf("missing_errno=%d\n", errno);
	return 0;
}

/* Keyed lookup-and-delete on a hash map: the value comes back, the entry is
 * gone, and a second call reports ENOENT. */
static int check_lookup_and_delete(void)
{
	int fd = map_create(BPF_MAP_TYPE_HASH, sizeof(unsigned int),
			    sizeof(unsigned long long), 16, 0);
	if (fd < 0) {
		printf("create_ok=0 create_errno=%d\n", errno);
		return 0;
	}
	unsigned int key = 1;
	unsigned long long value = 0xaabbccddeeff0011ULL;
	if (map_update(fd, &key, &value, BPF_ANY) != 0) {
		printf("update_ok=0 update_errno=%d\n", errno);
		return 0;
	}
	unsigned long long out = 0;
	errno = 0;
	int ret = map_lookup_and_delete(fd, &key, &out);
	printf("hash_ret=%d\n", ret);
	printf("hash_errno=%d\n", errno);
	printf("hash_value=%#llx\n", out);

	unsigned long long after = 0;
	errno = 0;
	ret = map_lookup(fd, &key, &after);
	printf("after_ret=%d\n", ret);
	printf("after_errno=%d\n", errno);

	errno = 0;
	ret = map_lookup_and_delete(fd, &key, &out);
	printf("again_ret=%d\n", ret);
	printf("again_errno=%d\n", errno);
	return 0;
}

/* Queue maps keep the keyless pop behavior. */
static int check_queue_pop(void)
{
	int fd = map_create(BPF_MAP_TYPE_QUEUE, 0, sizeof(unsigned long long),
			    16, 0);
	if (fd < 0) {
		printf("queue_create_ok=0 queue_create_errno=%d\n", errno);
		return 0;
	}
	unsigned long long value = 0x0123456789abcdefULL;
	if (map_update(fd, NULL, &value, BPF_ANY) != 0) {
		printf("queue_push_ok=0 queue_push_errno=%d\n", errno);
		return 0;
	}
	unsigned long long out = 0;
	errno = 0;
	int ret = map_lookup_and_delete(fd, NULL, &out);
	printf("queue_ret=%d\n", ret);
	printf("queue_errno=%d\n", errno);
	printf("queue_value=%#llx\n", out);
	errno = 0;
	ret = map_lookup_and_delete(fd, NULL, &out);
	printf("queue_empty_ret=%d\n", ret);
	printf("queue_empty_errno=%d\n", errno);
	return 0;
}

/* A loaded program keeps the maps its instructions reference alive, so a
 * loader may close its own map fd right after BPF_PROG_LOAD. The map slot must
 * not be handed out again until the program is closed. */
static int check_map_held_by_prog(void)
{
	int map_fd = map_create(BPF_MAP_TYPE_ARRAY, sizeof(unsigned int),
				sizeof(unsigned long long), 1, 0);
	if (map_fd < 0) {
		printf("create_ok=0 create_errno=%d\n", errno);
		return 0;
	}
	struct bpf_insn insns[4];
	memset(insns, 0, sizeof(insns));
	insns[0].code = BPF_LD | BPF_DW | BPF_IMM;
	insns[0].dst_reg = BPF_REG_1;
	insns[0].src_reg = BPF_PSEUDO_MAP_FD;
	insns[0].imm = map_fd;
	insns[2].code = BPF_ALU64 | BPF_MOV | BPF_K;
	insns[3].code = BPF_JMP | BPF_EXIT;
	union bpf_attr attr;
	memset(&attr, 0, sizeof(attr));
	attr.prog_type = BPF_PROG_TYPE_KPROBE;
	attr.insns = (unsigned long long)(unsigned long)insns;
	attr.insn_cnt = 4;
	attr.license = (unsigned long long)(unsigned long)"GPL";
	int prog_fd = bpf_call(BPF_PROG_LOAD, &attr);
	if (prog_fd < 0) {
		printf("load_ok=0 load_errno=%d\n", errno);
		return 0;
	}
	printf("close_map_ret=%d\n", close(map_fd));
	errno = 0;
	int again = close(map_fd);
	printf("close_again_ret=%d\n", again);
	printf("close_again_errno=%d\n", errno);
	int held_fd = map_create(BPF_MAP_TYPE_ARRAY, sizeof(unsigned int),
				 sizeof(unsigned long long), 1, 0);
	printf("reused_while_held=%d\n", held_fd == map_fd);
	printf("close_prog_ret=%d\n", close(prog_fd));
	int freed_fd = map_create(BPF_MAP_TYPE_ARRAY, sizeof(unsigned int),
				  sizeof(unsigned long long), 1, 0);
	printf("reused_after_prog_close=%d\n", freed_fd == map_fd);
	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 2) {
		return 64;
	}
	if (strcmp(argv[1], "lookup-errno") == 0) {
		return check_lookup_errno();
	}
	if (strcmp(argv[1], "lookup-and-delete") == 0) {
		return check_lookup_and_delete();
	}
	if (strcmp(argv[1], "queue-pop") == 0) {
		return check_queue_pop();
	}
	if (strcmp(argv[1], "map-held-by-prog") == 0) {
		return check_map_held_by_prog();
	}
	return 64;
}