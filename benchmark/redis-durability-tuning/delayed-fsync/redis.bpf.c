/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>

#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define EXTENDED_UFUNC_IOURING_INIT 1006
#define EXTENDED_UFUNC_IOURING_SUBMIT_WRITE 1007
#define EXTENDED_UFUNC_IOURING_SUBMIT_FSYNC 1008
#define EXTENDED_UFUNC_IOURING_WAIT_AND_SEEN 1009
#define EXTENDED_UFUNC_IOURING_SUBMIT 1010

static long (*io_uring_init_global)(void) = (void *)
    EXTENDED_UFUNC_IOURING_INIT;
static long (*io_uring_submit_write)(int fd, char *buf,
                                     unsigned long long size) = (void *)
    EXTENDED_UFUNC_IOURING_SUBMIT_WRITE;
static long (*io_uring_submit_fsync)(int fd) = (void *)
    EXTENDED_UFUNC_IOURING_SUBMIT_FSYNC;
static long (*io_uring_wait_and_seen)(void) = (void *)
    EXTENDED_UFUNC_IOURING_WAIT_AND_SEEN;
static long (*io_uring_submit)(void) = (void *)EXTENDED_UFUNC_IOURING_SUBMIT;

// unsigned long long last_ns = 0;
int successful_writeback_count = 0;
int current_pid = 0;

SEC("uprobe")
int BPF_UPROBE(start_config, int __fd) {
  bpf_printk("start called and init io_uring\n");
  io_uring_init_global();
  current_pid = (bpf_get_current_pid_tgid() >> 32);
  successful_writeback_count = -1;
  return 0;
}

SEC("uprobe")
int BPF_UPROBE(start_fsync, int __fd) {
  // bpf_printk("fsync called and submit io_uring\n");
  if (successful_writeback_count == 0) {
    // bpf_printk("fsync called and no writeback is completed");
    io_uring_wait_and_seen();
  }
  successful_writeback_count = 0;
  io_uring_submit_fsync(__fd);
  io_uring_submit();
  bpf_override_return(0, 0);
  return 0;
}

char LICENSE[] SEC("license") = "GPL";
