# FUSE Caching with bpftime

## Overview

The Filesystem in Userspace (FUSE) framework offers reliability and security advantages compared to in-kernel alternatives but imposes considerable runtime overhead due to additional context switches for every I/O system call.

While solutions like ExtFuse eliminate much of this overhead by enabling FUSE filesystems to push logic into the kernel, they require custom kernel modules that are difficult to maintain.

This project demonstrates how bpftime provides the same benefits without requiring custom kernel modules.

## Implementation

We implement two extensions that accelerate applications using FUSE:

### 1. Metadata Cache

This extension accelerates repeated lookups to the same filesystem entries by:

- Using bpftime's automated syscall tracepoints for `open`, `close`, `getdents`, and `stat`
- Building on bpftime's automated customizability extension class
- Adding function capabilities for bpftime helpers that interact with file paths (e.g., `realpath`)
- Using process-kernel shared maps
- Maintaining cache consistency through a kprobe extension on `unlink` in the kernel

### 2. Permission Checking Blacklist

This extension accelerates permission checking for functions accessing filesystem entries (e.g., `open`) by using bpftime's automated customizability extension class.

## Performance Evaluation

We evaluated performance improvements using bpftime to implement caching in FUSE on System A with FUSE file systems:

- **Passthrough**: Passes filesystem operations directly to the underlying file system
- **LoggedFS**: Logs all filesystem operations to a file before passing them to the underlying file system

### Workloads Tested

1. 100,000 `fstatat` calls to a file in the FUSE directory
2. 100,000 `openat` calls to a file in the FUSE directory
3. The Linux utility `find` traversing through the directory

