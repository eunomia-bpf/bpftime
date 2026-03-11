// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/**
 * multi_gpu_probe.c - Userspace loader for multi-GPU load balance eBPF probe
 *
 * Loads the eBPF probe and periodically prints a GPU-internal timing
 * dashboard showing:
 *   - Kernel invocation count
 *   - Block-level latency distribution histogram
 *   - Min / max / avg block duration
 *
 * This provides visibility into GPU-internal execution that host-side
 * CUDA events cannot observe.
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
#include <errno.h>
#include "./.output/multi_gpu_probe.skel.h"
#include <inttypes.h>

#define warn(...) fprintf(stderr, __VA_ARGS__)

struct gpu_stat {
	uint32_t grid_size;
	uint64_t time_sum;
	uint64_t block_count;
	double avg_ns;
};

static const char *BUCKET_LABELS[] = { "<1us",     "1-10us",  "10-100us",
					"100us-1ms", "1-10ms", "10-100ms",
					"100ms+" };
#define NUM_BUCKETS 7

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	if (level == LIBBPF_DEBUG)
		return 0;
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

static void sig_handler(int sig)
{
	exiting = true;
}

static uint64_t read_map_u64(int fd, uint32_t key)
{
	uint64_t val = 0;
	bpf_map_lookup_elem(fd, &key, &val);
	return val;
}

static void print_dashboard(struct multi_gpu_probe_bpf *obj)
{
	time_t t;
	struct tm *tm;
	char ts[32];
	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);

	// Read invocation count
	int fd_invoke = bpf_map__fd(obj->maps.invoke_count);
	uint64_t invocations = read_map_u64(fd_invoke, 0);

	// Read duration stats
	int fd_stats = bpf_map__fd(obj->maps.duration_stats);
	uint64_t sum_ns = read_map_u64(fd_stats, 0);
	uint64_t min_ns = read_map_u64(fd_stats, 1);
	uint64_t max_ns = read_map_u64(fd_stats, 2);
	uint64_t block_count = read_map_u64(fd_stats, 3);

	double avg_ns = (block_count > 0) ? (double)sum_ns / block_count : 0;

	// Read histogram
	int fd_hist = bpf_map__fd(obj->maps.latency_hist);
	uint64_t hist[NUM_BUCKETS];
	uint64_t hist_max = 0;
	for (int i = 0; i < NUM_BUCKETS; i++) {
		hist[i] = read_map_u64(fd_hist, (uint32_t)i);
		if (hist[i] > hist_max)
			hist_max = hist[i];
	}

	printf("\033[2J\033[H"); // clear screen
	printf("╔══════════════════════════════════════════════════"
	       "════════════════╗\n");
	printf("║  GPU-INTERNAL BLOCK LATENCY MONITOR (eBPF)  "
	       "   %s        ║\n",
	       ts);
	printf("╠══════════════════════════════════════════════════"
	       "════════════════╣\n");
	printf("║  Kernel invocations: %-10" PRIu64
	       "  Blocks profiled: %-12" PRIu64 " ║\n",
	       invocations, block_count);
	printf("╠══════════════════════════════════════════════════"
	       "════════════════╣\n");

	if (block_count > 0) {
		printf("║  Block Duration:  min=%-8" PRIu64
		       "ns  avg=%-10.0f"
		       "ns  max=%-8" PRIu64 "ns ║\n",
		       min_ns, avg_ns, max_ns);
	} else {
		printf("║  Block Duration:  (no data "
		       "yet)                                ║\n");
	}

	// Per-GPU stats (identified by gridDim.x)
	int fd_gpu_time = bpf_map__fd(obj->maps.per_gpu_time);
	int fd_gpu_count = bpf_map__fd(obj->maps.per_gpu_count);

	struct gpu_stat gpu_info[16];
	int num_gpus = 0;

	uint32_t prev_key, cur_key;
	if (bpf_map_get_next_key(fd_gpu_time, NULL, &cur_key) == 0) {
		do {
			uint64_t gtime = 0, gcnt = 0;
			bpf_map_lookup_elem(fd_gpu_time, &cur_key, &gtime);
			bpf_map_lookup_elem(fd_gpu_count, &cur_key, &gcnt);
			gpu_info[num_gpus].grid_size = cur_key;
			gpu_info[num_gpus].time_sum = gtime;
			gpu_info[num_gpus].block_count = gcnt;
			gpu_info[num_gpus].avg_ns =
				(gcnt > 0) ? (double)gtime / gcnt : 0;
			num_gpus++;
			prev_key = cur_key;
		} while (num_gpus < 16 &&
			 bpf_map_get_next_key(fd_gpu_time, &prev_key,
					      &cur_key) == 0);
	}

	// Sort by grid_size (ascending = GPU order)
	for (int i = 0; i < num_gpus - 1; i++) {
		for (int j = 0; j < num_gpus - i - 1; j++) {
			if (gpu_info[j].grid_size >
			    gpu_info[j + 1].grid_size) {
				struct gpu_stat tmp = gpu_info[j];
				gpu_info[j] = gpu_info[j + 1];
				gpu_info[j + 1] = tmp;
			}
		}
	}

	if (num_gpus > 0) {
		printf("╠══════════════════════════════════════════════════"
		       "════════════════╣\n");
		printf("║  Per-GPU Block Timing (by grid "
		       "size):                          ║\n");

		double max_avg = 0;
		for (int i = 0; i < num_gpus; i++) {
			if (gpu_info[i].avg_ns > max_avg)
				max_avg = gpu_info[i].avg_ns;
		}

		for (int i = 0; i < num_gpus; i++) {
			float pct = (max_avg > 0) ?
					    gpu_info[i].avg_ns / max_avg * 100 :
					    0;
			int bar_len = (int)(pct / 10);
			char bar[12];
			for (int j = 0; j < 10; j++)
				bar[j] = (j < bar_len) ? '#' : '.';
			bar[10] = '\0';

			printf("║  Grid %5u │ avg %8.0f ns │ "
			       "%6" PRIu64 " blks │ %s %5.1f%%  ║\n",
			       gpu_info[i].grid_size, gpu_info[i].avg_ns,
			       gpu_info[i].block_count, bar, pct);
		}
	}

	printf("╠══════════════════════════════════════════════════"
	       "════════════════╣\n");
	printf("║  Latency Histogram (per-block "
	       "distribution):                     ║\n");

	for (int i = 0; i < NUM_BUCKETS; i++) {
		int bar_len =
			(hist_max > 0) ? (int)(hist[i] * 30 / hist_max) : 0;
		char bar[32];
		for (int j = 0; j < 30; j++)
			bar[j] = (j < bar_len) ? '#' : ' ';
		bar[30] = '\0';

		printf("║  %10s │%s│ %-8" PRIu64 "         ║\n",
		       BUCKET_LABELS[i], bar, hist[i]);
	}

	printf("╚══════════════════════════════════════════════════"
	       "════════════════╝\n");
	printf("\n  These metrics are measured INSIDE the GPU via eBPF "
	       "globaltimer.\n");
	printf("  Host-side CUDA events cannot observe per-block "
	       "latency distribution.\n");
	printf("  Press Ctrl-C to exit.\n");
	fflush(stdout);
}

int main(int argc, char **argv)
{
	struct multi_gpu_probe_bpf *skel;
	int err;

	libbpf_set_print(libbpf_print_fn);

	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	skel = multi_gpu_probe_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	err = multi_gpu_probe_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton\n");
		goto cleanup;
	}

	err = multi_gpu_probe_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}

	printf("Multi-GPU eBPF probe loaded.\n");
	printf("Run multi_gpu_vec_add with bpftime to see traces.\n\n");

	while (!exiting) {
		print_dashboard(skel);
		sleep(2);
	}

cleanup:
	multi_gpu_probe_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
