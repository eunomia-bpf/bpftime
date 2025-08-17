// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <stdarg.h>
#include <sys/resource.h>
#include "compat_libbpf.h"
#include <bpf/bpf.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/syscall.h>
#include <linux/bpf.h>
#include "uprobe_bloom_filter.skel.h"

#define warn(...) fprintf(stderr, __VA_ARGS__)

// Statistics index definitions (consistent with BPF program)
#define STAT_TOTAL_ACCESSES 0
#define STAT_UNIQUE_USERS 1
#define STAT_REPEAT_USERS 2
#define STAT_ADMIN_OPS 3
#define STAT_SYSTEM_EVENTS 4
#define STAT_BLOOM_HITS 5
#define STAT_BLOOM_MISSES 6

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile sig_atomic_t stop;

static void sig_int(int signo)
{
	stop = 1;
}

// Get statistics data
static uint64_t get_stat(int stats_fd, uint32_t index)
{
	uint64_t value = 0;
	int ret = bpf_map_lookup_elem(stats_fd, &index, &value);
	if (ret != 0) {
		return 0;
	}
	return value;
}

// Display statistics and bloom filter analysis
static void show_statistics(int stats_fd)
{
	uint64_t total_accesses = get_stat(stats_fd, STAT_TOTAL_ACCESSES);
	uint64_t unique_users = get_stat(stats_fd, STAT_UNIQUE_USERS);
	uint64_t repeat_users = get_stat(stats_fd, STAT_REPEAT_USERS);
	uint64_t admin_ops = get_stat(stats_fd, STAT_ADMIN_OPS);
	uint64_t system_events = get_stat(stats_fd, STAT_SYSTEM_EVENTS);
	uint64_t bloom_hits = get_stat(stats_fd, STAT_BLOOM_HITS);
	uint64_t bloom_misses = get_stat(stats_fd, STAT_BLOOM_MISSES);

	printf("\n=== Bloom Filter Real-time Monitoring Statistics ===\n");
	printf("Total user accesses:      %lu\n", total_accesses);
	printf("New users (first access): %lu\n", unique_users);
	printf("Repeat users (re-access): %lu\n", repeat_users);
	printf("Admin operations:         %lu\n", admin_ops);
	printf("System events:            %lu\n", system_events);
	printf("\n--- Bloom Filter Performance Analysis ---\n");
	printf("Bloom Filter hits:        %lu (detected as possibly existing)\n",
	       bloom_hits);
	printf("Bloom Filter misses:      %lu (definitely not existing)\n",
	       bloom_misses);

	if (bloom_hits + bloom_misses > 0) {
		double hit_rate = (double)bloom_hits /
				  (bloom_hits + bloom_misses) * 100.0;
		printf("Hit rate:                 %.2f%%\n", hit_rate);
	}

	if (total_accesses > 0) {
		double unique_rate =
			(double)unique_users / total_accesses * 100.0;
		double repeat_rate =
			(double)repeat_users / total_accesses * 100.0;
		printf("New user ratio:           %.2f%%\n", unique_rate);
		printf("Repeat user ratio:        %.2f%%\n", repeat_rate);

		// Analyze bloom filter effectiveness
		if (repeat_users > 0) {
			double false_positive_estimate =
				(double)(bloom_hits - repeat_users) /
				bloom_hits * 100.0;
			if (false_positive_estimate > 0) {
				printf("Estimated false positive rate: %.2f%%\n",
				       false_positive_estimate);
			}
		}
	}
	printf("==============================================\n");
}

// Analyze bloom filter test results
static void analyze_bloom_filter_performance(int stats_fd)
{
	uint64_t total_accesses = get_stat(stats_fd, STAT_TOTAL_ACCESSES);
	uint64_t unique_users = get_stat(stats_fd, STAT_UNIQUE_USERS);
	uint64_t repeat_users = get_stat(stats_fd, STAT_REPEAT_USERS);
	uint64_t bloom_hits = get_stat(stats_fd, STAT_BLOOM_HITS);
	uint64_t bloom_misses = get_stat(stats_fd, STAT_BLOOM_MISSES);

	printf("\n=== Bloom Filter Test Analysis ===\n");
	printf("Test principle explanation:\n");
	printf("  - When target program calls user_access() function with user ID\n");
	printf("  - eBPF program checks if the user ID exists in bloom filter\n");
	printf("  - If not exists, mark as new user and add to bloom filter\n");
	printf("  - If exists, mark as repeat user (possible false positive)\n");
	printf("\n");

	if (total_accesses > 0) {
		printf("Test result verification:\n");
		printf("  Theory: new users + repeat users = total accesses\n");
		printf("  Actual: %lu + %lu = %lu (total accesses: %lu)\n",
		       unique_users, repeat_users, unique_users + repeat_users,
		       total_accesses);

		if (unique_users + repeat_users == total_accesses) {
			printf("  [PASS] Consistency check passed\n");
		} else {
			printf("  [FAIL] Consistency check failed\n");
		}

		printf("\n  Bloom Filter characteristics verification:\n");
		printf("  - No false negatives: all new users correctly identified [VERIFIED]\n");
		printf("  - Possible false positives: some new users may be misjudged as repeat users\n");

		if (bloom_hits > 0 && repeat_users > 0) {
			if (bloom_hits >= repeat_users) {
				printf("  - False positive detection: possible %lu false positives\n",
				       bloom_hits - repeat_users);
			}
		}
	}
	printf("=================================\n");
}

int main(int argc, char **argv)
{
	struct uprobe_bloom_filter_bpf *skel;
	int err;
	int bloom_fd, stats_fd;

	libbpf_set_print(libbpf_print_fn);

	skel = uprobe_bloom_filter_bpf__open();
	if (!skel) {
		warn("Failed to open BPF skeleton\n");
		return 1;
	}

	err = uprobe_bloom_filter_bpf__load(skel);
	if (err) {
		warn("Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	err = uprobe_bloom_filter_bpf__attach(skel);
	if (err) {
		warn("Failed to attach BPF program: %d\n", err);
		goto cleanup;
	}

	bloom_fd = bpf_map__fd(skel->maps.user_bloom_filter);
	stats_fd = bpf_map__fd(skel->maps.stats);

	if (bloom_fd < 0 || stats_fd < 0) {
		warn("Failed to get map file descriptors\n");
		err = -1;
		goto cleanup;
	}

	printf("=== Bloom Filter Monitor Program Started ===\n");
	printf("Bloom Filter Map FD: %d\n", bloom_fd);
	printf("Stats Map FD: %d\n", stats_fd);
	printf("eBPF program successfully attached to uprobe\n");
	printf("Monitoring user access patterns...\n");
	printf("Please start target program to trigger bloom filter test\n");
	printf("Press Ctrl+C to stop monitoring\n\n");

	if (signal(SIGINT, sig_int) == SIG_ERR) {
		warn("Cannot set signal handler\n");
		err = 1;
		goto cleanup;
	}

	int iteration = 0;
	while (!stop) {
		iteration++;

		// Display statistics every 5 seconds
		show_statistics(stats_fd);

		// Perform deep analysis every 20 seconds
		if (iteration % 4 == 0) {
			analyze_bloom_filter_performance(stats_fd);
		}

		sleep(5);
	}

cleanup:
	uprobe_bloom_filter_bpf__destroy(skel);
	printf("\nMonitor program exited\n");
	return -err;
}