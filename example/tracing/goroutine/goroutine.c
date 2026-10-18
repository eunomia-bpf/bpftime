// SPDX-License-Identifier: GPL-2.0
#include <argp.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include "goroutine.h"
#include "goroutine.skel.h"

static struct env {
	bool verbose;
	long min_duration_ms;
} env;

const char *argp_program_version = "goroutine 0.0";
const char *argp_program_bug_address = "<bpf@vger.kernel.org>";
const char argp_program_doc[] =
"BPF goroutine demo application.\n"
"\n"
"It traces process start and exits and shows associated \n"
"information (filename, process duration, PID and PPID, etc).\n"
"\n"
"USAGE: ./goroutine [-d <min-duration-ms>] [-v]\n";

static const struct argp_option opts[] = {
	{ "verbose", 'v', NULL, 0, "Verbose debug output" },
	{ "duration", 'd', "DURATION-MS", 0, "Minimum process duration (ms) to report" },
	{},
};

static error_t parse_arg(int key, char *arg, struct argp_state *state)
{
	switch (key) {
	case 'v':
		env.verbose = true;
		break;
	case 'd':
		errno = 0;
		env.min_duration_ms = strtol(arg, NULL, 10);
		if (errno || env.min_duration_ms <= 0) {
			fprintf(stderr, "Invalid duration: %s\n", arg);
			argp_usage(state);
		}
		break;
	case ARGP_KEY_ARG:
		argp_usage(state);
		break;
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

static const struct argp argp = {
	.options = opts,
	.parser = parse_arg,
	.doc = argp_program_doc,
};

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	if (level == LIBBPF_DEBUG && !env.verbose)
		return 0;
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

static void sig_handler(int sig)
{
	exiting = true;
}

static int handle_event(void *ctx, void *data, size_t data_sz)
{
	const struct goroutine_execute_data *e = data;
	struct tm *tm;
	char ts[32];
	time_t t;

	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);

	// enum goroutine_state {
	// IDLE,
	// RUNNABLE,
	// RUNNING,
	// SYSCALL,
	// WAITING,
	// MORIBUND_UNUSED,
	// DEAD,
	// ENQUEUE_UNUSED,
	// COPYSTACK,
	// PREEMPTED,
	// };

	// struct goroutine_execute_data {
	// enum goroutine_state state;
	// unsigned long goid;
	// int pid;
	// int tgid;
	// };

	printf("%-8s  ",
		       ts);
	// print the state and goroutine_execute_data in one line
	// print the states string
	switch (e->state) {
		case IDLE:
			printf("%-5s", "IDLE");
			break;
		case RUNNABLE:
			printf("%-5s", "RUNNABLE");
			break;
		case RUNNING:
			printf("%-5s", "RUNNING");
			break;
		case SYSCALL:
			printf("%-5s", "SYSCALL");
			break;
		case WAITING:
			printf("%-5s", "WAITING");
			break;
		case MORIBUND_UNUSED:
			printf("%-5s", "MORIBUND_UNUSED");
			break;
		case DEAD:
			printf("%-5s", "DEAD");
			break;
		case ENQUEUE_UNUSED:
			printf("%-5s", "ENQUEUE_UNUSED");
			break;
		case COPYSTACK:
			printf("%-5s", "COPYSTACK");
			break;
		case PREEMPTED:
			printf("%-5s", "PREEMPTED");
			break;
		default:
			printf("%-5s", "UNKNOWN");
			break;
	}
	printf("%-16s %-4lu %-7d %-7d\n",
		       "  Goid", e->goid, e->pid, e->tgid);
	return 0;
}

int main(int argc, char **argv)
{
	struct ring_buffer *rb = NULL;
	struct goroutine_bpf *skel;
	int err;

	/* Parse command line arguments */
	err = argp_parse(&argp, argc, argv, 0, NULL, NULL);
	if (err)
		return err;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = goroutine_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = goroutine_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}

	/* Attach tracepoints */
	err = goroutine_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}

	/* Set up ring buffer polling */
	rb = ring_buffer__new(bpf_map__fd(skel->maps.rb), handle_event, NULL, NULL);
	if (!rb) {
		err = -1;
		fprintf(stderr, "Failed to create ring buffer\n");
		goto cleanup;
	}

	/* Process events */
	printf("%-8s %-5s %-16s %-7s %-7s %s\n",
	       "TIME", "EVENT", "COMM", "PID", "PPID", "FILENAME/EXIT CODE");
	while (!exiting) {
		err = ring_buffer__poll(rb, 100 /* timeout, ms */);
		/* Ctrl-C will cause -EINTR */
		if (err == -EINTR) {
			err = 0;
			break;
		}
		if (err < 0) {
			printf("Error polling perf buffer: %d\n", err);
			break;
		}
	}

cleanup:
	/* Clean up */
	ring_buffer__free(rb);
	goroutine_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
