// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/* Copyright (c) 2020 Facebook */
#include <string.h>
#include <signal.h>
#include <stdio.h>
#include <sys/types.h>
#include <time.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <stdlib.h>
#include ".output/minimal_go_uprobe.skel.h"
#include <inttypes.h>
#include <argp.h>
#define warn(...) fprintf(stderr, __VA_ARGS__)

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

static void sig_handler(int sig)
{
	exiting = true;
}

struct {
	pid_t target_pid;
	char exe_path[256];
	int set;
} env;

static const struct argp_option opts[] = {
	{ "pid", 'p', "PID", 0, "PID of the process to trace" },
	{ "path", 'd', "PATH", 0, "Path of the executable path" },
	{},
};

static error_t parse_arg(int key, char *arg, struct argp_state *state)
{
	switch (key) {
	case 'p':
		if (env.set) {
			fprintf(stderr, "You may only set one of pid or path");
			return ARGP_ERR_UNKNOWN;
		}

		errno = 0;
		env.target_pid = strtol(arg, NULL, 10);
		if (errno || env.target_pid <= 0) {
			fprintf(stderr, "Invalid pid: %s\n", arg);
			argp_usage(state);
		}
		env.set = 1;
		break;
	case 'd':
		if (env.set) {
			fprintf(stderr, "You may only set one of pid or path");
			return ARGP_ERR_UNKNOWN;
		}
		strcpy(env.exe_path, arg);
		env.set = 2;
		break;
	case ARGP_KEY_ARG:
		argp_usage(state);
		break;
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

const char argp_program_doc[] =
	"Go uprobe demo.\n"
	"\n"
	"It traces runtime.casgstatus.\n"
	"Specify the target pid through -p, or executable path through -d";

static const struct argp argp = {
	.options = opts,
	.parser = parse_arg,
	.doc = argp_program_doc,
};
int main(int argc, char **argv)
{
	struct minimal_go_uprobe_bpf *skel;
	int err;

	err = argp_parse(&argp, argc, argv, 0, NULL, NULL);
	if (err)
		return err;
	if (!env.set) {
		fprintf(stderr, "You must specify one of -d or -p\n");
		return 1;
	}
	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = minimal_go_uprobe_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = minimal_go_uprobe_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}
	LIBBPF_OPTS(bpf_uprobe_opts, attach_opts,
		    .func_name = "runtime.casgstatus", .retprobe = false);
	struct bpf_link *attach;
	if (env.set == 1) {
		sprintf(env.exe_path, "/proc/%d/exe", env.target_pid);
		// Use PID
		attach = bpf_program__attach_uprobe_opts(
			skel->progs.go_trace_test, env.target_pid, env.exe_path,
			0, &attach_opts);
	} else if (env.set == 2) {
		attach = bpf_program__attach_uprobe_opts(
			skel->progs.go_trace_test, -1, env.exe_path, 0,
			&attach_opts);
	} else {
		fprintf(stderr, "You must specify one of -d or -p");
		goto cleanup;
	}
	if (!attach) {
		fprintf(stderr, "Failed to attach BPF skeleton: %d\n", errno);
		err = -1;
		goto cleanup;
	}
	while (!exiting) {
		sleep(1);
		puts("See /sys/kernel/tracing/trace_pipe for output");
		fflush(stdout);
	}
cleanup:
	/* Clean up */
	minimal_go_uprobe_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
