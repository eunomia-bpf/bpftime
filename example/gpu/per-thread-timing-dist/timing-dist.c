#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <stdlib.h>
#include ".output/timing-dist.skel.h"

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

struct summary_t {
	uint64_t min;
	uint64_t max;
	uint64_t sum;
	uint64_t count;
};

static void print_summary(struct bpf_map *s)
{
	int fd = bpf_map__fd(s);
	uint32_t key = 0;
	struct summary_t val;
	int err;

	err = bpf_map_lookup_elem(fd, &key, &val);
	if (err) {
		fprintf(stderr, "bpf_map_lookup_elem failed for summary: %s\n",
			strerror(errno));
		return;
	}

	printf("Summary:\n");
	printf("  Min: %" PRIu64 " ns\n", val.min == UINT64_MAX ? 0 : val.min);
	printf("  Max: %" PRIu64 " ns\n", val.max);
	printf("  Avg: %" PRIu64 " ns\n",
	       val.count > 0 ? val.sum / val.count : 0);
	printf("  Total threads: %" PRIu64 "\n", val.count);
}

static void print_log2_hist(struct bpf_map *h)
{
	int err, fd = bpf_map__fd(h);
	uint32_t i;

	printf("log2(nanoseconds):\n");

	for (i = 0; i < 64; i++) {
		uint64_t count;
		err = bpf_map_lookup_elem(fd, &i, &count);
		if (err) {
			fprintf(stderr,
				"bpf_map_lookup_elem failed for key %d: %s\n",
				i, strerror(errno));
			return;
		}
		if (count > 0) {
			if (i == 63) {
				printf("  [%" PRIu64 ", ...): %" PRIu64 "\n",
				       1ULL << i, count);
			} else {
				printf("  [%" PRIu64 ", %" PRIu64 "): %" PRIu64
				       "\n",
				       1ULL << i, 1ULL << (i + 1), count);
			}
		}
	}
}

int main(int argc, char **argv)
{
	struct timing_dist_bpf *skel;
	int err;

	libbpf_set_print(libbpf_print_fn);

	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	skel = timing_dist_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	err = timing_dist_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}

	err = timing_dist_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}

	// Initialize summary map
	{
		int summary_fd = bpf_map__fd(skel->maps.summary_map);
		uint32_t key = 0;
		struct summary_t init_val = {
			.min = UINT64_MAX, .max = 0, .sum = 0, .count = 0
		};
		err = bpf_map_update_elem(summary_fd, &key, &init_val, BPF_ANY);
		if (err) {
			fprintf(stderr,
				"Failed to initialize summary map: %s\n",
				strerror(errno));
			goto cleanup;
		}
	}

	printf("Successfully started! Please run `timed_work_kernel` in another terminal.\n");
	printf("BPF program attached. Tracing execution time of timed_work_kernel.\n");

	while (!exiting) {
		sleep(2);
		printf("\n--- Timing Distribution ---\n");
		print_log2_hist(skel->maps.timing_dist);
		print_summary(skel->maps.summary_map);
	}

cleanup:
	timing_dist_bpf__destroy(skel);
	return -err;
}
