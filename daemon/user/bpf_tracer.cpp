// Description: bpf_tracer daemon
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
#include "bpf_tracer_event.h"
#include "bpf_tracer.skel.h"
#include "daemon_config.hpp"
#include "handle_bpf_event.hpp"
#include "daemon.hpp"
#include <cassert>
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>

#define NSEC_PER_SEC 1000000000ULL

using namespace bpftime;

static volatile sig_atomic_t exiting = 0;
static bool verbose = false;

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	if (level == LIBBPF_DEBUG && !verbose)
		return 0;
	return vfprintf(stderr, format, args);
}

static void sig_int(int signo)
{
	exiting = 1;
}

static int handle_event_rb(void *ctx, void *data, size_t data_sz)
{
	const struct event *e = (const struct event *)data;
	bpf_event_handler *handler = (bpf_event_handler *)ctx;
	assert(handler != NULL);
	return handler->handle_event(e);
}

int bpftime::start_daemon(struct daemon_config env)
{
	LIBBPF_OPTS(bpf_object_open_opts, open_opts);
	struct ring_buffer *rb = NULL;
	struct bpf_tracer_bpf *obj = NULL;
	int err;

	spdlog::cfg::load_env_levels();

	libbpf_set_print(libbpf_print_fn);

	if (signal(SIGINT, sig_int) == SIG_ERR) {
		fprintf(stderr, "can't set signal handler: %s\n",
			strerror(errno));
		err = 1;
		return err;
	}

	obj = bpf_tracer_bpf__open();
	if (!obj) {
		err = -1;
		fprintf(stderr, "failed to open BPF object\n");
		return err;
	}

	/* initialize global data (filtering options) */
	obj->rodata->target_pid = env.pid;
	obj->rodata->enable_replace_prog = env.enable_replace_prog;
	strncpy(obj->rodata->new_uprobe_path, env.new_uprobe_path, PATH_LENTH);
	obj->rodata->enable_replace_uprobe = env.enable_replace_uprobe;
	obj->rodata->uprobe_perf_type = determine_uprobe_perf_type();
	obj->rodata->kprobe_perf_type = determine_kprobe_perf_type();
	obj->rodata->submit_bpf_events = env.submit_bpf_events;

	if (!env.show_open) {
		bpf_program__set_autoload(
			obj->progs.tracepoint__syscalls__sys_exit_open, false);
		bpf_program__set_autoload(
			obj->progs.tracepoint__syscalls__sys_enter_open, false);
		bpf_program__set_autoload(
			obj->progs.tracepoint__syscalls__sys_exit_openat,
			false);
		bpf_program__set_autoload(
			obj->progs.tracepoint__syscalls__sys_enter_openat,
			false);
	}

	bpftime_driver driver(env, obj);
	// update handler config
	bpf_event_handler handler = bpf_event_handler(env, driver);
	verbose = env.verbose;

	err = bpf_tracer_bpf__load(obj);
	if (err) {
		fprintf(stderr, "failed to load BPF object: %d\n", err);
		goto cleanup;
	}

	err = bpf_tracer_bpf__attach(obj);
	if (err) {
		fprintf(stderr, "failed to attach BPF programs\n");
		goto cleanup;
	}

	/* Set up ring buffer polling */
	rb = ring_buffer__new(bpf_map__fd(obj->maps.rb), handle_event_rb,
			      &handler, NULL);
	if (!rb) {
		err = -1;
		fprintf(stderr, "Failed to create ring buffer\n");
		goto cleanup;
	}

	/* main: poll */
	while (!exiting) {
		err = ring_buffer__poll(rb, 100 /* timeout, ms */);
		if (err < 0 && err != -EINTR) {
			spdlog::error("error polling perf buffer: {}",
				      strerror(-err));
			// goto cleanup;
		}
		/* reset err to return 0 if exiting */
		err = 0;
	}

cleanup:
	ring_buffer__free(rb);
	bpf_tracer_bpf__destroy(obj);

	return err != 0;
}
