#ifndef BPFTIME_DAEMON_CONFIG_HPP
#define BPFTIME_DAEMON_CONFIG_HPP

#include <unistd.h>

struct env {
	pid_t pid;
	uid_t uid;
	bool verbose;
	bool failed;
	bool show_open;
};

#endif // BPFTIME_DAEMON_CONFIG_HPP
