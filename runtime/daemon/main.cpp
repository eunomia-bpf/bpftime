// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
// Copyright (c) 2019 Facebook
// Copyright (c) 2020 Netflix
//
// Based on bpf_mocker(8) from BCC by Brendan Gregg and others.
// 14-Feb-2020   Brendan Gregg   Created this.
#include <argp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "daemon.hpp"

using namespace bpftime;

static struct env env = { .uid = static_cast<uid_t>(-1) };

const char *argp_program_version = "bpftime-daemon 0.1";
const char *argp_program_bug_address = "https://github.com/eunomia-bpf/bpftime";
const char argp_program_doc[] = "Trace and modify bpf syscalls\n";

static const struct argp_option opts[] = {
	{ "pid", 'p', "PID", 0, "Process ID to trace" },
	{ "uid", 'u', "UID", 0, "User ID to trace" },
	{ "open", 'o', "OPEN", 0, "Show open events" },
	{ "verbose", 'v', NULL, 0, "Verbose debug output" },
	{ "failed", 'x', NULL, 0, "Failed opens only" },
	{},
};

static error_t parse_arg(int key, char *arg, struct argp_state *state)
{
	static int pos_args;
	long int pid, uid;

	switch (key) {
	case 'v':
		env.verbose = true;
		break;
	case 'x':
		env.failed = true;
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

	err = argp_parse(&argp, argc, argv, 0, NULL, NULL);
	if (err)
		return err;

	return start_daemon(env);
}
