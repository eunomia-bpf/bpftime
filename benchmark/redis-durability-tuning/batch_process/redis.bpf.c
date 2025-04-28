/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <vmlinux.h>
#include "rocksdb.h"
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define EXTENDED_HELPER_IOURING_INIT  1006
#define EXTENDED_HELPER_IOURING_SUBMIT_WRITE 1007
#define EXTENDED_HELPER_IOURING_SUBMIT_FSYNC 1008
#define EXTENDED_HELPER_IOURING_WAIT_AND_SEEN 1009
#define EXTENDED_HELPER_IOURING_SUBMIT 1010

static long (*io_uring_init_global)(void) = (void *) EXTENDED_HELPER_IOURING_INIT;
static long (*io_uring_submit_write)(int fd, char *buf, unsigned long long size) = (void *) EXTENDED_HELPER_IOURING_SUBMIT_WRITE;
static long (*io_uring_submit_fsync)(int fd) = (void *) EXTENDED_HELPER_IOURING_SUBMIT_FSYNC;
static long (*io_uring_wait_and_seen)(void) = (void *) EXTENDED_HELPER_IOURING_WAIT_AND_SEEN;
static long (*io_uring_submit)(void) = (void *) EXTENDED_HELPER_IOURING_SUBMIT;

int patch_size = 48;
int current_count = 0;

SEC("uprobe//root/zys/bpftime-evaluation/redis/src/redis-server:aofWrite")
int BPF_UPROBE(write, int __fd, const void *__buf, unsigned long long __n) {
    if (current_count < patch_size) {
      io_uring_submit_write(__fd, __buf, __n);
      current_count++;
      
    } else {
      io_uring_submit_write(__fd, __buf, __n);
      current_count++;
      io_uring_submit();
      for (int i = 0; i < current_count; i++) {
        io_uring_wait_and_seen();
      }
      current_count = 0;
    }
    bpf_override_return(0, __n);
  // bpf_printk("write called");
  // bpf_override_return(0, 5);
  return 0;
}

SEC("uprobe//lib/x86_64-linux-gnu/libc.so.6:fdatasync")
int BPF_UPROBE(fsync, int __fd) {
  if (current_count < patch_size) {
      io_uring_submit_fsync(__fd);
      current_count++;
    } else {
      io_uring_submit_fsync(__fd);
      current_count++;
      io_uring_submit();
      for (int i = 0; i < current_count; i++) {
        io_uring_wait_and_seen();
      }
      current_count = 0;
    }
  bpf_override_return(0, 0);
  return 0;
}

SEC("uprobe//root/zys/bpftime-evaluation/redis/src/redis-server:initServerConfig")
int BPF_UPROBE(start, int __fd) {
  bpf_printk("start called and init io_uring\n");
  io_uring_init_global();
  return 0;
}

char LICENSE[] SEC("license") = "GPL";
