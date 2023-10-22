#ifndef BPFTIME_DAEMON_CONFIG_HPP
#define BPFTIME_DAEMON_CONFIG_HPP

#include <unistd.h>

struct env {
	// the target pid of eBPF application to trace
	pid_t pid;
	// the target uid of eBPF application to trace
	uid_t uid;
	// print verbose debug output
	bool verbose;
	// print open syscalls (default: false)
	bool show_open;
};

#endif // BPFTIME_DAEMON_CONFIG_HPP
