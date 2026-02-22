// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/*
 * host_map_test.c - Userspace program to test GPU Host Maps
 *
 * This program loads the BPF program and periodically reads
 * the Host-backed GPU maps to display statistics.
 */
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "./.output/host_map_test.skel.h"

#define warn(...) fprintf(stderr, __VA_ARGS__)

// Default GPU thread count for PERGPUTD_ARRAY_HOST_MAP
// This should match the actual number of threads launched
#define GPU_THREAD_COUNT (1<<20)

// Default number of entries for host maps (must match BPF code)
#ifndef HOST_MAP_MAX_ENTRIES
#define HOST_MAP_MAX_ENTRIES 10
#endif

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

static int print_stat(struct host_map_test_bpf *obj)
{
	time_t t;
	struct tm *tm;
	char ts[16];
	int err = 0;
	uint64_t value;

	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);

	printf("%-9s\n", ts);

	// Print shared_counter map (BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP)
	int shared_fd = bpf_map__fd(obj->maps.shared_counter);
	printf("  [shared_counter map (GPU_ARRAY_HOST_MAP)]\n");

	// Use get_next_key to iterate through all keys with data
	uint32_t shared_key, *shared_prev_key = NULL;
	int has_data = 0;
	while (true) {
		err = bpf_map_get_next_key(shared_fd, shared_prev_key, &shared_key);
		if (err) {
			if (errno == ENOENT) {
				break;  // No more keys
			}
			warn("bpf_map_get_next_key failed: %s\n", strerror(errno));
			break;
		}
		err = bpf_map_lookup_elem(shared_fd, &shared_key, &value);
		if (err == 0 && value != 0) {
			printf("    key=%-2" PRIu32 " value: %" PRIu64 "\n", shared_key, value);
			has_data = 1;
		}
		shared_prev_key = &shared_key;
	}
	if (!has_data) {
		printf("    (no data yet)\n");
	}

	// Print perthread_counter map (BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP)
	// Note: PERGPUTD_ARRAY returns value_size * thread_count bytes per key
	int pt_fd = bpf_map__fd(obj->maps.perthread_counter);
	printf("  [perthread_counter map (PERGPUTD_ARRAY_HOST_MAP)]\n");
	static uint64_t pt_values[GPU_THREAD_COUNT];  // Buffer for all thread values
	const char *key_names[] = { "call_count", "exec_time_ns", "thread_id" };

	// Use get_next_key to iterate through all keys with data
	uint32_t pt_key, *pt_prev_key = NULL;
	int has_pt_data = 0;
	while (true) {
		err = bpf_map_get_next_key(pt_fd, pt_prev_key, &pt_key);
		if (err) {
			if (errno == ENOENT) {
				break;  // No more keys
			}
			warn("bpf_map_get_next_key failed: %s\n", strerror(errno));
			break;
		}
		memset(pt_values, 0, sizeof(pt_values));
		err = bpf_map_lookup_elem(pt_fd, &pt_key, pt_values);
		if (err == 0) {
			// Sum all thread values
			uint64_t total = 0;
			int active_count = 0;
			for (int t = 0; t < GPU_THREAD_COUNT; t++) {
				if (pt_values[t] != 0) {
					total += pt_values[t];
					active_count++;
				}
			}
			// Only print if there's actual data
			if (total > 0) {
				const char *key_name = (pt_key < 3) ? key_names[pt_key] : "unknown";
				printf("    key=%-2" PRIu32 " (%s): total=%" PRIu64 " (from %d threads)\n",
				       pt_key, key_name, total, active_count);
				has_pt_data = 1;
			}
		}
		pt_prev_key = &pt_key;
	}
	if (!has_pt_data) {
		printf("    (no data yet)\n");
	}

	// Print thread_timestamp map (also PERGPUTD_ARRAY_HOST_MAP)
	int ts_fd = bpf_map__fd(obj->maps.thread_timestamp);
	printf("  [thread_timestamp map (PERGPUTD_ARRAY_HOST_MAP)]\n");
	static uint64_t ts_values[GPU_THREAD_COUNT];  // Buffer for all thread values
	memset(ts_values, 0, sizeof(ts_values));
	uint32_t ts_key = 0;
	err = bpf_map_lookup_elem(ts_fd, &ts_key, ts_values);
	if (err == 0) {
		// Count how many threads have non-zero timestamps
		int active_count = 0;
		for (int t = 0; t < GPU_THREAD_COUNT; t++) {
			if (ts_values[t] != 0) {
				active_count++;
			}
		}
		printf("    key=0: %d threads have timestamps\n", active_count);
	}

	fflush(stdout);
	return 0;
}

int main(int argc, char **argv)
{
	struct host_map_test_bpf *skel;
	int err;

	printf("Host Map Test - Testing GPU Host-backed Maps\n");
	printf("============================================\n");
	printf("\nMap types being tested:\n");
	printf("  - BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP (1513)\n");
	printf("      Shared storage in Host memory, all threads share same values\n");
	printf("  - BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP (1512)\n");
	printf("      Per-thread storage in Host memory, each thread has its own slot\n");

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = host_map_test_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = host_map_test_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}

	err = host_map_test_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}

	printf("\nBPF program loaded and attached. Waiting for CUDA kernel events...\n");
	printf("Run vec_add with bpftime agent in another terminal.\n");
	printf("Press Ctrl-C to exit.\n\n");

	while (!exiting) {
		sleep(1);
		print_stat(skel);
	}

cleanup:
	/* Clean up */
	host_map_test_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
