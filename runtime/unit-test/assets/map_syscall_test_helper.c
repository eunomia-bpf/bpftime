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
	return 64;
}