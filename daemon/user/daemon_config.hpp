#ifndef BPFTIME_DAEMON_CONFIG_HPP
#define BPFTIME_DAEMON_CONFIG_HPP

#include <unistd.h>

struct daemon_config {
	// the target pid of eBPF application to trace
	pid_t pid;
	// the target uid of eBPF application to trace
	uid_t uid;
	// print verbose debug output
	bool verbose;
	// print open syscalls (default: false)
	bool show_open;
	bool enable_replace_prog;
	bool enable_replace_uprobe;
};

#endif // BPFTIME_DAEMON_CONFIG_HPP
