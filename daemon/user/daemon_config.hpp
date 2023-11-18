/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef BPFTIME_DAEMON_CONFIG_HPP
#define BPFTIME_DAEMON_CONFIG_HPP

#include <cstdint>
#include <set>
#include <unistd.h>
#include <string>
#include <vector>

#define PATH_LENTH 255

// configuration for bpftime daemon
struct daemon_config {
	// the target pid of eBPF application to trace
	pid_t pid = 0;
	// the target uid of eBPF application to trace
	uid_t uid = 0;
	// print verbose debug output
	bool verbose = false;
	// print open syscalls (default: false)
	// Open syscall may related to bpf config, so we need to handle it
	bool show_open = false;
	// enable replace prog to support bypass kernel verifier
	bool enable_replace_prog = false;
	// enable replace uprobe to make kernel uprobe not break user space
	// uprobe
	bool enable_replace_uprobe = true;
	// use the new uprobe path to replace the old one in original syscall
	char new_uprobe_path[PATH_LENTH] = "\0";
	// bpftime cli path for bpftime daemon to create prog and link, maps
	std::string bpftime_cli_path = "~/.bpftime/bpftime";
	// bpftime tool path for bpftime daemon to run bpftime
	std::string bpftime_tool_path = "~/.bpftime/bpftimetool";
	// should bpftime be involve
	bool is_driving_bpftime = true;
	// should trace and submit bpf related detail events
	bool submit_bpf_events = false;
	// specify whether uprobe should work similar to kernel uprobe and auto
	// attach to the target process
	bool enable_auto_attach = false;
	// minimal duration of a process to be traced by uprobe
	// skip short lived process to reduce overhead
	int duration_ms = 1000;
	// Only uprobes in the list will be run in userspace
	std::set<uint64_t> whitelist_uprobes;
	bool whitelist_enabled() const
	{
		return !whitelist_uprobes.empty();
	}
};

#endif // BPFTIME_DAEMON_CONFIG_HPP
