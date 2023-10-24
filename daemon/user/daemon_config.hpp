#ifndef BPFTIME_DAEMON_CONFIG_HPP
#define BPFTIME_DAEMON_CONFIG_HPP

#include <unistd.h>
#include <string>

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
	// enable replace uprobe to make kernel uprobe not break user space uprobe
	bool enable_replace_uprobe = true;
	char new_uprobe_path[PATH_LENTH] = "\0";
	// bpftime cli path for bpftime daemon to create prog and link, maps
	std::string bpftime_cli_path = "~/.bpftime/bpftime";
	// bpftime tool path for bpftime daemon to run bpftime
	std::string bpftime_tool_path = "~/.bpftime/bpftimetool";
	// should bpftime be involve
	bool is_driving_bpftime = true;
};

#endif // BPFTIME_DAEMON_CONFIG_HPP
