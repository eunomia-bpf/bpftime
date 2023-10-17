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
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <linux/perf_event.h>
#include "bpf-mocker-event.h"
#include "bpf-mocker.skel.h"

/* Tune the buffer size and wakeup rate. These settings cope with roughly
 * 50k opens/sec.
 */
#define PERF_BUFFER_PAGES 64
#define PERF_BUFFER_TIME_MS 10

/* Set the poll timeout when no events occur. This can affect -d accuracy. */
#define PERF_POLL_TIMEOUT_MS 100

#define NSEC_PER_SEC 1000000000ULL

static volatile sig_atomic_t exiting = 0;

static struct env {
	pid_t pid;
	uid_t uid;
	bool verbose;
	bool failed;
	bool show_open;
} env = { .uid = INVALID_UID };

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
		if (errno || uid < 0 || uid >= INVALID_UID) {
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

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	if (level == LIBBPF_DEBUG && !env.verbose)
		return 0;
	return vfprintf(stderr, format, args);
}

static void sig_int(int signo)
{
	exiting = 1;
}

static int handle_open_events(const struct event *e)
{
	struct tm *tm;
	char ts[32];
	time_t t;
	int fd, err;
	if (!env.show_open) {
		return 0;
	}

	/* prepare fields */
	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);
	if (e->open_data.ret >= 0) {
		fd = e->open_data.ret;
		err = 0;
	} else {
		fd = -1;
		err = -e->open_data.ret;
	}

	/* print output */
	printf("OPEN %-6d %-16s %3d %3d ", e->pid, e->comm, fd, err);
	printf("%s\n", e->open_data.fname);
	return 0;
}

const char *bpf_cmd_strings[] = {
	"BPF_MAP_CREATE",	 "BPF_MAP_LOOKUP_ELEM",
	"BPF_MAP_UPDATE_ELEM",	 "BPF_MAP_DELETE_ELEM",
	"BPF_MAP_GET_NEXT_KEY",	 "BPF_PROG_LOAD",
	"BPF_OBJ_PIN",		 "BPF_OBJ_GET",
	"BPF_PROG_ATTACH",	 "BPF_PROG_DETACH",
	"BPF_PROG_TEST_RUN",	 "BPF_PROG_GET_NEXT_ID",
	"BPF_MAP_GET_NEXT_ID",	 "BPF_PROG_GET_FD_BY_ID",
	"BPF_MAP_GET_FD_BY_ID",	 "BPF_OBJ_GET_INFO_BY_FD",
	"BPF_PROG_QUERY",	 "BPF_RAW_TRACEPOINT_OPEN",
	"BPF_BTF_LOAD",		 "BPF_BTF_GET_FD_BY_ID",
	"BPF_TASK_FD_QUERY",	 "BPF_MAP_LOOKUP_AND_DELETE_ELEM",
	"BPF_MAP_FREEZE",	 "BPF_BTF_GET_NEXT_ID",
	"BPF_MAP_LOOKUP_BATCH",	 "BPF_MAP_LOOKUP_AND_DELETE_BATCH",
	"BPF_MAP_UPDATE_BATCH",	 "BPF_MAP_DELETE_BATCH",
	"BPF_LINK_CREATE",	 "BPF_LINK_UPDATE",
	"BPF_LINK_GET_FD_BY_ID", "BPF_LINK_GET_NEXT_ID",
	"BPF_ENABLE_STATS",	 "BPF_ITER_CREATE",
	"BPF_LINK_DETACH",	 "BPF_PROG_BIND_MAP",
};

const char* bpf_prog_type_strings[] = {
    "BPF_PROG_TYPE_UNSPEC",
    "BPF_PROG_TYPE_SOCKET_FILTER",
    "BPF_PROG_TYPE_KPROBE",
    "BPF_PROG_TYPE_SCHED_CLS",
    "BPF_PROG_TYPE_SCHED_ACT",
    "BPF_PROG_TYPE_TRACEPOINT",
    "BPF_PROG_TYPE_XDP",
    "BPF_PROG_TYPE_PERF_EVENT",
    "BPF_PROG_TYPE_CGROUP_SKB",
    "BPF_PROG_TYPE_CGROUP_SOCK",
    "BPF_PROG_TYPE_LWT_IN",
    "BPF_PROG_TYPE_LWT_OUT",
    "BPF_PROG_TYPE_LWT_XMIT",
    "BPF_PROG_TYPE_SOCK_OPS",
    "BPF_PROG_TYPE_SK_SKB",
    "BPF_PROG_TYPE_CGROUP_DEVICE",
    "BPF_PROG_TYPE_SK_MSG",
    "BPF_PROG_TYPE_RAW_TRACEPOINT",
    "BPF_PROG_TYPE_CGROUP_SOCK_ADDR",
    "BPF_PROG_TYPE_LWT_SEG6LOCAL",
    "BPF_PROG_TYPE_LIRC_MODE2",
    "BPF_PROG_TYPE_SK_REUSEPORT",
    "BPF_PROG_TYPE_FLOW_DISSECTOR",
    "BPF_PROG_TYPE_CGROUP_SYSCTL",
    "BPF_PROG_TYPE_RAW_TRACEPOINT_WRITABLE",
    "BPF_PROG_TYPE_CGROUP_SOCKOPT",
    "BPF_PROG_TYPE_TRACING",
    "BPF_PROG_TYPE_STRUCT_OPS",
    "BPF_PROG_TYPE_EXT",
    "BPF_PROG_TYPE_LSM",
    "BPF_PROG_TYPE_SK_LOOKUP",
    "BPF_PROG_TYPE_SYSCALL",
};

static int handle_bpf_event(const struct event *e)
{
	struct tm *tm;
	char ts[32];
	time_t t;

	/* prepare fields */
	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);

	const char *cmd_str = e->bpf_data.bpf_cmd >= (sizeof(bpf_cmd_strings) /
						  sizeof(bpf_cmd_strings[0])) ?
				  "UNKNOWN COMMAND" :
				  bpf_cmd_strings[e->bpf_data.bpf_cmd];

	/* print output */
	printf("BPF %-6d %-16s %-16s %d\n", e->pid, e->comm, cmd_str, e->bpf_data.ret);
	return 0;
}

#define PERF_TYPE_MAX_ID 16
const char* perf_type_id_strings[PERF_TYPE_MAX_ID] = {
    [PERF_TYPE_HARDWARE]     = "PERF_TYPE_HARDWARE",
    [PERF_TYPE_SOFTWARE]     = "PERF_TYPE_SOFTWARE",
    [PERF_TYPE_TRACEPOINT]   = "PERF_TYPE_TRACEPOINT",
    [PERF_TYPE_HW_CACHE]     = "PERF_TYPE_HW_CACHE",
    [PERF_TYPE_RAW]          = "PERF_TYPE_RAW",
    [PERF_TYPE_BREAKPOINT]   = "PERF_TYPE_BREAKPOINT",
};

static int handle_perf_event(const struct event *e)
{
	struct tm *tm;
	char ts[32];
	time_t t;

	/* prepare fields */
	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);

	const char *type_id_str = e->perf_event_data.attr.type >= (sizeof(perf_type_id_strings) /
						  sizeof(perf_type_id_strings[0])) ?
				  "UNKNOWN TYPE" :
				  perf_type_id_strings[e->perf_event_data.attr.type];

	/* print output */
	printf("PERF %-6d %-16s %-8s %d\n", e->pid, e->comm, type_id_str, e->perf_event_data.pid);
	return 0;
}


/*
 * this function is expected to parse integer in the range of [0, 2^31-1] from
 * given file using scanf format string fmt. If actual parsed value is
 * negative, the result might be indistinguishable from error
 */
static int parse_uint_from_file(const char *file, const char *fmt)
{
	int err, ret;
	FILE *f;

	f = fopen(file, "re");
	if (!f) {
		err = -errno;
		fprintf(stderr, "failed to open '%s\n", file);
		return err;
	}
	err = fscanf(f, fmt, &ret);
	if (err != 1) {
		err = err == EOF ? -EIO : -errno;
		fprintf(stderr,"failed to parse '%s'\n", file);
		fclose(f);
		return err;
	}
	fclose(f);
	return ret;
}

static int determine_kprobe_perf_type(void)
{
	const char *file = "/sys/bus/event_source/devices/kprobe/type";

	return parse_uint_from_file(file, "%d\n");
}

static int determine_uprobe_perf_type(void)
{
	const char *file = "/sys/bus/event_source/devices/uprobe/type";

	return parse_uint_from_file(file, "%d\n");
}


static int handle_load_bpf_prog_event(const struct event *e)
{
	struct tm *tm;
	char ts[32];
	time_t t;

	/* prepare fields */
	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);


	const char *prog_type_str = e->bpf_loaded_prog.type >= (sizeof(bpf_prog_type_strings) /
						  sizeof(bpf_prog_type_strings[0])) ?
				  "UNKNOWN PROG TYPE" :
				  bpf_prog_type_strings[e->bpf_loaded_prog.type];
	
	/* print output */
	printf("LOAD %-6d %-16s %-16s %-16s %d\n", e->pid, e->comm, prog_type_str, e->bpf_loaded_prog.prog_name, e->bpf_loaded_prog.insn_cnt);
	return 0;
}

static int handle_event_rb(void *ctx, void *data, size_t data_sz)
{
	const struct event *e = (const struct event *)data;
	switch (e->type) {
	case SYS_OPEN:
		return handle_open_events(e);
		break;
	case SYS_PERF_EVENT_OPEN:
		return handle_perf_event(e);
		break;
	case SYS_BPF:
		return handle_bpf_event(e);
		break;
	case BPF_PROG_LOAD_EVENT:
		return handle_load_bpf_prog_event(e);
		break;
	}
	return 0;
}

void handle_lost_events(void *ctx, int cpu, __u64 lost_cnt)
{
	fprintf(stderr, "Lost %llu events on CPU #%d!\n", lost_cnt, cpu);
}

int main(int argc, char **argv)
{
	LIBBPF_OPTS(bpf_object_open_opts, open_opts);
	static const struct argp argp = {
		.options = opts,
		.parser = parse_arg,
		.doc = argp_program_doc,
	};
	// struct perf_buffer *pb = NULL;
	struct ring_buffer *rb = NULL;
	struct bpf_mocker_bpf *obj = NULL;
	int err;

	libbpf_set_print(libbpf_print_fn);
	err = argp_parse(&argp, argc, argv, 0, NULL, NULL);
	if (err)
		return err;

	obj = bpf_mocker_bpf__open();
	if (!obj) {
		fprintf(stderr, "failed to open BPF object\n");
		goto cleanup;
	}

	/* initialize global data (filtering options) */
	obj->rodata->target_pid = env.pid;
	obj->rodata->disable_modify = true;

	perf_type_id_strings[determine_uprobe_perf_type()] = "PERF_TYPE_UPROBE";
	perf_type_id_strings[determine_kprobe_perf_type()] = "PERF_TYPE_KPROBE";

	err = bpf_mocker_bpf__load(obj);
	if (err) {
		fprintf(stderr, "failed to load BPF object: %d\n", err);
		goto cleanup;
	}

	err = bpf_mocker_bpf__attach(obj);
	if (err) {
		fprintf(stderr, "failed to attach BPF programs\n");
		goto cleanup;
	}
	/* print headers */
	printf("%-6s %-16s %3s %3s ", "PID", "COMM", "FD", "ERR");
	printf("%s", "PATH");
	printf("\n");

	/* Set up ring buffer polling */
	rb = ring_buffer__new(bpf_map__fd(obj->maps.rb), handle_event_rb, NULL,
			      NULL);
	if (!rb) {
		err = -1;
		fprintf(stderr, "Failed to create ring buffer\n");
		goto cleanup;
	}

	if (signal(SIGINT, sig_int) == SIG_ERR) {
		fprintf(stderr, "can't set signal handler: %s\n",
			strerror(errno));
		err = 1;
		goto cleanup;
	}

	/* main: poll */
	while (!exiting) {
		err = ring_buffer__poll(rb, 100 /* timeout, ms */);
		if (err < 0 && err != -EINTR) {
			fprintf(stderr, "error polling perf buffer: %s\n",
				strerror(-err));
			goto cleanup;
		}
		/* reset err to return 0 if exiting */
		err = 0;
	}

cleanup:
	ring_buffer__free(rb);
	bpf_mocker_bpf__destroy(obj);

	return err != 0;
}
