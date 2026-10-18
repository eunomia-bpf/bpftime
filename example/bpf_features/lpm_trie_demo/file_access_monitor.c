// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
/* bpftime LPM Trie file access monitor */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "file_access_filter.skel.h"

// Type definitions
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t s32;

// Structures matching BPF program
struct lpm_key {
	u32 prefixlen;
	char data[64];
};

struct event_data {
	u64 timestamp;
	u32 pid;
	u32 tid;
	u32 counter;
	u32 function_id;
	s32 flags;
	char filename[64];
	char comm[16];
	u32 allowed;
};

static volatile bool exiting = false;

static void sig_handler(int sig)
{
	exiting = true;
}

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

int main(int argc, char **argv)
{
	struct file_access_filter_bpf *skel;
	int err;

	printf("bpftime LPM Trie file access monitor\n");
	printf("====================================\n\n");

	// Set up libbpf errors and debug info callback
	libbpf_set_print(libbpf_print_fn);

	// Bump RLIMIT_MEMLOCK to allow BPF sub-system to do anything
	struct rlimit rlim_new = {
		.rlim_cur = RLIM_INFINITY,
		.rlim_max = RLIM_INFINITY,
	};

	if (setrlimit(RLIMIT_MEMLOCK, &rlim_new)) {
		// Try with a smaller limit if INFINITY fails
		rlim_new.rlim_cur = 512 * 1024 * 1024; // 512MB
		rlim_new.rlim_max = 512 * 1024 * 1024;
		if (setrlimit(RLIMIT_MEMLOCK, &rlim_new)) {
			printf("Warning: Failed to increase RLIMIT_MEMLOCK limit, continuing anyway\n");
		} else {
			printf("Set RLIMIT_MEMLOCK to 512MB\n");
		}
	} else {
		printf("Set RLIMIT_MEMLOCK to unlimited\n");
	}

	// Clean up handler
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	// Open and load BPF application
	skel = file_access_filter_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	// Load & verify BPF programs
	err = file_access_filter_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton: %d\n",
			err);
		goto cleanup;
	}

	printf("BPF program loaded successfully\n");

	// Initialize LPM Trie with allowed path prefixes
	printf("Initializing LPM Trie with allowed paths...\n");

	// Add allowed path prefixes
	const char *allowed_prefixes[] = { "/tmp/", "/var/tmp/", "/usr/share/",
					   "/home/user/documents/" };

	u32 allow_policy = 1; // 1 = allowed

	for (int i = 0; i < 4; i++) {
		struct lpm_key key = {};
		int len = strlen(allowed_prefixes[i]);

		// Copy prefix to key
		strncpy(key.data, allowed_prefixes[i], sizeof(key.data) - 1);
		key.data[sizeof(key.data) - 1] = '\0';

		// Set prefix length in bits
		key.prefixlen = len * 8;

		// Insert into LPM Trie
		int ret = bpf_map_update_elem(
			bpf_map__fd(skel->maps.allowed_paths), &key,
			&allow_policy, BPF_ANY);
		if (ret != 0) {
			fprintf(stderr,
				"Failed to insert prefix '%s' into LPM Trie: %d\n",
				allowed_prefixes[i], ret);
		} else {
			printf("Added allowed prefix: %s (len=%d bits)\n",
			       allowed_prefixes[i], key.prefixlen);
		}
	}

	// Auto-attach BPF program using skeleton
	printf("Attaching BPF program...\n");
	err = file_access_filter_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF program: %d\n", err);
		goto cleanup;
	}

	printf("BPF program attached successfully\n");
	printf("Monitoring file access events...\n");
	printf("Press Ctrl+C to stop\n\n");

	// Main event loop
	while (!exiting) {
		struct event_data event;

		// Try to pop event from queue
		int ret = bpf_map_lookup_and_delete_elem(
			bpf_map__fd(skel->maps.events_queue), NULL, &event);

		if (ret == 0) {
			// Process event
			printf("Event #%u: %s[%u] accessed '%s' (flags=0x%x) -> %s\n",
			       event.counter, event.comm, event.pid,
			       event.filename, event.flags,
			       event.allowed ? "ALLOWED" : "DENIED");
		} else {
			// No events, sleep briefly
			usleep(100000); // 100ms
		}
	}

	printf("\nShutting down...\n");

	// Print final statistics
	u32 key = 0;
	u32 total_calls = 0;
	int ret = bpf_map_lookup_elem(bpf_map__fd(skel->maps.call_counter),
				      &key, &total_calls);
	if (ret == 0) {
		printf("Total function calls monitored: %u\n", total_calls);
	}

cleanup:
	file_access_filter_bpf__destroy(skel);
	printf("Cleanup completed\n");
	return err < 0 ? -err : 0;
}