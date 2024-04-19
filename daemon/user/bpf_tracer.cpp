/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "spdlog/common.h"
#include <argp.h>
#include <cstddef>
#include <filesystem>
#include <fstream>
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
#include <map>
#include <spdlog/cfg/env.h>
#include <chrono>

#define NSEC_PER_SEC 1000000000ULL

using namespace bpftime;

static volatile sig_atomic_t exiting = 0;
static bool verbose = false;

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	if (level == LIBBPF_DEBUG && !verbose)
		return 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
	return vfprintf(stderr, format, args);
#pragma GCC diagnostic pop
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

// Function to get current time in nanoseconds using std::chrono
long long get_current_time_ns() {
    auto now = std::chrono::steady_clock::now();
    auto ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
    return ns.time_since_epoch().count();
}


static int process_exec_maps(bpf_event_handler *handler, bpf_tracer_bpf *obj,
			     daemon_config &env)
{
	if (!obj || obj->maps.exec_start == NULL) {
		return 0;
	}
	event e;
	int pid = 0, next_pid = 0;

	static std::map<int, event> pid_map;

	std::map<int, event> new_pid_map = {};
	std::map<int, event> remain_pid_map = pid_map;

	if (bpf_map__get_next_key(obj->maps.exec_start, NULL, &next_pid,
				  sizeof(pid)) != 0) {
		return 0;
	}
	do {
		pid = next_pid;
		int res = bpf_map__lookup_elem(obj->maps.exec_start, &pid,
					       sizeof(pid), &e, sizeof(e), 0);
		if (res != 0) {
			continue;
		}
		
		long long current_nanoseconds;
		long start_time_ms;
		// Get current time in nanoseconds
    	current_nanoseconds = get_current_time_ns();

		start_time_ms =
			(current_nanoseconds - e.exec_data.time_ns) / 1000000;
		if (start_time_ms < env.duration_ms) {
			// ignore short-lived processes
			return 0;
		}
		// find exec new process and exit process
		new_pid_map[pid] = e;
		if (remain_pid_map.find(pid) == remain_pid_map.end()) {
			// new pid
			handle_event_rb(handler, &e, sizeof(e));
		} else {
			remain_pid_map.erase(pid);
		}
	} while (bpf_map__get_next_key(obj->maps.exec_start, &pid, &next_pid,
				       sizeof(pid)) == 0);
	// remain pid is exit process, not occur in the map
	for (auto pid : remain_pid_map) {
		e = pid.second;
		e.exec_data.exit_event = 1;
		handle_event_rb(handler, &e, sizeof(e));
	}
	pid_map = new_pid_map;
	return 0;
}
extern "C" const char placeholder[];
extern "C" size_t placeholder_len;
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

	// strncpy(obj->rodata->new_uprobe_path, env.new_uprobe_path,
	// PATH_LENTH);
	// TODO: currently using `/a` as the replacing executable path to uprobe
	// perf event in the kernel, since long strings (such as bpftime_daemon
	// it self) may break userspace memory. Find a better way to solve this
	// in the future
	strncpy(obj->rodata->new_uprobe_path, "/a", PATH_LENTH);
	std::filesystem::path elf_path("/a");
	if (!std::filesystem::exists(elf_path)) {
		SPDLOG_INFO("Releasing: {}", elf_path.c_str());
		std::ofstream ofs(elf_path, std::ios::binary);
		if (!ofs.is_open()) {
			SPDLOG_ERROR("Unable to open {} for writting",
				     elf_path.c_str());
			return 1;
		}
		ofs.write(placeholder, placeholder_len);

	} else {
		SPDLOG_INFO("{} exists", elf_path.c_str());
	}
	obj->rodata->enable_replace_uprobe = env.enable_replace_uprobe;
	obj->rodata->uprobe_perf_type = determine_uprobe_perf_type();
	obj->rodata->kprobe_perf_type = determine_kprobe_perf_type();
	obj->rodata->submit_bpf_events = env.submit_bpf_events;
	obj->rodata->current_pid = getpid();
	obj->rodata->enable_whitelist = env.whitelist_enabled();
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

	if (!env.enable_auto_attach) {
		bpf_program__set_autoload(obj->progs.handle_exit, false);
		bpf_program__set_autoload(obj->progs.handle_exec, false);
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
	if (env.whitelist_enabled()) {
		SPDLOG_INFO("userspace uprobe whitelist enabled");
		int whitelist_map_fd =
			bpf_map__fd(obj->maps.whitelist_hook_addr);
		for (uint64_t addr : env.whitelist_uprobes) {
			SPDLOG_INFO("Whitelist: {:x}", addr);
			uint32_t dummy = 0;
			int err = bpf_map_update_elem(whitelist_map_fd, &addr,
						      &dummy, 0);
			if (err < 0) {
				SPDLOG_ERROR(
					"Unable to update whitelist map, err={}",
					err);
			}
		}
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
		err = ring_buffer__poll(rb, 300 /* timeout, ms */);
		if (err < 0 && err != -EINTR) {
			SPDLOG_ERROR("error polling perf buffer: {}",
				     strerror(-err));
			// goto cleanup;
		}
		process_exec_maps(&handler, obj, env);
		/* reset err to return 0 if exiting */
		err = 0;
	}

cleanup:
	ring_buffer__free(rb);
	bpf_tracer_bpf__destroy(obj);

	return err != 0;
}
