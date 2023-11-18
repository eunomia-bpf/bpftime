/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <argp.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "daemon.hpp"
using namespace bpftime;

static struct daemon_config env = { .uid = static_cast<uid_t>(-1) };

const char *argp_program_version = "bpftime-daemon 0.1";
const char *argp_program_bug_address = "https://github.com/eunomia-bpf/bpftime";
const char argp_program_doc[] = "Trace and modify bpf syscalls\n";

static const struct argp_option opts[] = {
	{ "pid", 'p', "PID", 0, "Process ID to trace" },
	{ "uid", 'u', "UID", 0, "User ID to trace" },
	{ "open", 'o', "OPEN", 0, "Show open events" },
	{ "verbose", 'v', NULL, 0, "Verbose debug output" },
	{ "whitelist-uprobe", 'w', "UPROBE_ADDR", 0,
	  "Whitelist uprobe function addresses" },
	{},
};

static error_t parse_arg(int key, char *arg, struct argp_state *state)
{
	static int pos_args;
	long int pid, uid;
	uint64_t addr;
	switch (key) {
	case 'v':
		env.verbose = true;
		break;
	case 'o':
		env.show_open = true;
		break;
	case 'p':
		errno = 0;
		pid = strtol(arg, NULL, 10);
		if (errno || pid <= 0) {
			fprintf(stderr, "Invalid PID: %s\n", arg);
			argp_usage(state);
		}
		env.pid = pid;
		break;
	case 'u':
		errno = 0;
		uid = strtol(arg, NULL, 10);
		if (errno || uid < 0 || uid >= -1) {
			fprintf(stderr, "Invalid UID %s\n", arg);
			argp_usage(state);
		}
		env.uid = uid;
		break;
	case 'w':
		errno = 0;
		addr = strtoul(arg, nullptr, 0);
		if (errno) {
			std::cerr << "Invalid function address: " << arg
				  << std::endl;
			argp_usage(state);
		}
		env.whitelist_uprobes.insert(addr);
		break;
	case ARGP_KEY_ARG:
		if (pos_args++) {
			fprintf(stderr,
				"Unrecognized positional argument: %s\n", arg);
			argp_usage(state);
		}
		errno = 0;
		break;
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

int main(int argc, char **argv)
{
	static const struct argp argp = {
		.options = opts,
		.parser = parse_arg,
		.doc = argp_program_doc,
	};
	int err;
	// use current path as default path
	strncpy(env.new_uprobe_path, argv[0], PATH_LENTH);
	err = argp_parse(&argp, argc, argv, 0, NULL, NULL);
	if (err)
		return err;
	return start_daemon(env);
}
