// SM/Warp/Lane Mapping - Userspace Loader
// Displays GPU thread-to-hardware mapping information

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
#include <dlfcn.h>
#include "./.output/threadscheduling.skel.h"
#include <inttypes.h>

#define warn(...) fprintf(stderr, __VA_ARGS__)
#define MAX_SMS 128

struct thread_map {
	uint64_t sm_id;
	uint64_t warp_id;
	uint64_t lane_id;
	uint64_t block_x;
	uint64_t block_y;
	uint64_t block_z;
	uint64_t thread_x;
	uint64_t thread_y;
	uint64_t thread_z;
	uint64_t timestamp;
};

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

static void print_header(void)
{
	printf("\n");
	printf("╔════════════════════════════════════════════════════════════════════╗\n");
	printf("║              SM / Warp / Lane Mapping Report                       ║\n");
	printf("╚════════════════════════════════════════════════════════════════════╝\n");
}

/* Number of SMs the device physically has, or a negative value when the CUDA
 * driver cannot be reached. The kernel-side histogram only ever records SMs
 * that ran at least one sampled thread, so the observed id range cannot stand
 * in for the device width: a single-SM run would otherwise look perfect.
 * Resolved through dlopen so this loader keeps working without CUDA. */
static int query_physical_sm_count(void)
{
	static int cached = -2;
	if (cached != -2)
		return cached;
	cached = -1;

	void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
	if (!lib)
		lib = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
	if (!lib)
		return cached;

	typedef int (*cu_init_fn)(unsigned int);
	typedef int (*cu_device_get_fn)(int *, int);
	typedef int (*cu_device_get_attr_fn)(int *, int, int);
	cu_init_fn cu_init = (cu_init_fn)dlsym(lib, "cuInit");
	cu_device_get_fn cu_device_get =
		(cu_device_get_fn)dlsym(lib, "cuDeviceGet");
	cu_device_get_attr_fn cu_device_get_attr =
		(cu_device_get_attr_fn)dlsym(lib, "cuDeviceGetAttribute");

	/* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CUDA_SUCCESS == 0. */
	const int cuda_success = 0;
	const int attr_sm_count = 16;
	int device = 0;
	int count = 0;

	if (cu_init && cu_device_get && cu_device_get_attr &&
	    cu_init(0) == cuda_success && cu_device_get(&device, 0) == cuda_success &&
	    cu_device_get_attr(&count, attr_sm_count, device) == cuda_success &&
	    count > 0)
		cached = count;

	dlclose(lib);
	return cached;
}

static void print_sm_histogram(struct threadscheduling_bpf *obj)
{
	int fd = bpf_map__fd(obj->maps.sm_histogram);
	uint32_t key, *prev_key = NULL;
	uint64_t value;
	int err;

	// Collect SM data
	uint64_t sm_counts[MAX_SMS] = {0};
	uint64_t max_count = 0;
	uint64_t total_threads = 0;
	int max_sm = -1;

	while (1) {
		err = bpf_map_get_next_key(fd, prev_key, &key);
		if (err) {
			if (errno == ENOENT)
				break;
			return;
		}
		err = bpf_map_lookup_elem(fd, &key, &value);
		if (err)
			return;

		if (key < MAX_SMS) {
			sm_counts[key] = value;
			total_threads += value;
			if (value > max_count)
				max_count = value;
			if ((int)key > max_sm)
				max_sm = key;
		}
		prev_key = &key;
	}

	if (max_sm < 0) {
		printf("\n  No SM data collected yet.\n");
		return;
	}

	printf("\n┌─ SM Utilization Histogram ─────────────────────────────────────────┐\n");
	printf("│                                                                    │\n");

	// Print histogram bars
	const int bar_width = 40;
	for (int i = 0; i <= max_sm; i++) {
		if (sm_counts[i] > 0) {
			int bar_len = (max_count > 0) ?
				(int)((sm_counts[i] * bar_width) / max_count) : 0;
			if (bar_len == 0 && sm_counts[i] > 0)
				bar_len = 1;

			printf("│  SM %2d: ", i);
			for (int j = 0; j < bar_len; j++)
				printf("█");
			for (int j = bar_len; j < bar_width; j++)
				printf(" ");
			printf("    %6" PRIu64 " threads │\n", sm_counts[i]);
		}
	}

	printf("│                                                                    │\n");
	int active_sms = 0;
	for (int i = 0; i <= max_sm; i++)
		if (sm_counts[i] > 0)
			active_sms++;

	const int physical_sms = query_physical_sm_count();
	int sm_total = physical_sms > 0 ? physical_sms : max_sm + 1;
	// The histogram map holds MAX_SMS entries, so a wider device is measured
	// over the range the map can record rather than over an unpopulated tail.
	if (sm_total > MAX_SMS)
		sm_total = MAX_SMS;
	printf("│  Total threads: %-8" PRIu64 "  Active SMs: %-3d of %-3d      │\n",
	       total_threads, active_sms, sm_total);
	printf("└────────────────────────────────────────────────────────────────────┘\n");

	// Calculate load balance score against the physical SM count. An SM that
	// ran no sampled thread counts as fully idle, which is what makes a
	// single-SM run score low instead of perfect. The per-SM ideal is kept in
	// floating point: with fewer threads than SMs the integer division used
	// to floor it to zero and pin the score at a fixed 50%.
	if (total_threads > 0) {
		double ideal_per_sm = (double)total_threads / (double)sm_total;
		// SM identifiers need not be contiguous. Count idle physical SMs,
		// not the unused identifiers between sampled SMs.
		double total_deviation = ideal_per_sm * (sm_total - active_sms);
		for (int i = 0; i <= max_sm; i++) {
			if (sm_counts[i] == 0)
				continue;
			double diff = (double)sm_counts[i] - ideal_per_sm;
			total_deviation += diff < 0 ? -diff : diff;
		}
		double balance_score =
			100.0 * (1.0 - total_deviation / (2.0 * (double)total_threads));
		if (balance_score < 0)
			balance_score = 0;
		printf("\n  Load Balance Score: %.1f%% (100%% = perfect distribution%s)\n",
		       balance_score,
		       physical_sms > MAX_SMS ? ", over histogram SM range" :
		       physical_sms > 0 ? ", over all physical SMs" : ", over observed SMs");
	}
}

static void print_warp_distribution(struct threadscheduling_bpf *obj)
{
	int fd = bpf_map__fd(obj->maps.warp_histogram);
	uint32_t key, *prev_key = NULL;
	uint64_t value;
	int err;

	printf("\n┌─ Warp Distribution per SM ─────────────────────────────────────────┐\n");
	printf("│  SM   │ Warp ID │ Thread Count                                     │\n");
	printf("├───────┼─────────┼──────────────────────────────────────────────────┤\n");

	int count = 0;
	while (1) {
		err = bpf_map_get_next_key(fd, prev_key, &key);
		if (err) {
			if (errno == ENOENT)
				break;
			return;
		}
		err = bpf_map_lookup_elem(fd, &key, &value);
		if (err)
			return;

		uint32_t sm_id = (key >> 16) & 0xFFFF;
		uint32_t warp_id = key & 0xFFFF;
		printf("│  %3u  │   %3u   │ %8" PRIu64 "                                         │\n",
		       sm_id, warp_id, value);

		prev_key = &key;
		count++;
		if (count >= 20) {
			printf("│  ...  │   ...   │ (showing first 20 entries)                       │\n");
			break;
		}
	}

	if (count == 0) {
		printf("│       │         │ No warp data collected yet                       │\n");
	}

	printf("└───────┴─────────┴──────────────────────────────────────────────────┘\n");
}

static void print_thread_samples(struct threadscheduling_bpf *obj)
{
	int fd = bpf_map__fd(obj->maps.thread_mapping);
	uint32_t key, *prev_key = NULL;
	struct thread_map info;
	int err;

	printf("\n┌─ Thread-to-Hardware Mapping Samples ───────────────────────────────┐\n");
	printf("│  Block(x,y,z)  │ Thread(x,y,z) │  SM  │ Warp │ Lane │              │\n");
	printf("├────────────────┼───────────────┼──────┼──────┼──────┼──────────────┤\n");

	int count = 0;
	while (1) {
		err = bpf_map_get_next_key(fd, prev_key, &key);
		if (err) {
			if (errno == ENOENT)
				break;
			return;
		}
		err = bpf_map_lookup_elem(fd, &key, &info);
		if (err)
			return;

		printf("│  (%3" PRIu64 ",%2" PRIu64 ",%2" PRIu64 ")   │  (%3" PRIu64 ",%2" PRIu64 ",%2" PRIu64 ")  │ %4" PRIu64 " │ %4" PRIu64 " │ %4" PRIu64 " │              │\n",
		       info.block_x, info.block_y, info.block_z,
		       info.thread_x, info.thread_y, info.thread_z,
		       info.sm_id, info.warp_id, info.lane_id);

		prev_key = &key;
		count++;
		if (count >= 10) {
			printf("│  ...           │ ...           │ ...  │ ...  │ ...  │ (10 samples) │\n");
			break;
		}
	}

	if (count == 0) {
		printf("│                │               │      │      │      │ No data yet  │\n");
	}

	printf("└────────────────┴───────────────┴──────┴──────┴──────┴──────────────┘\n");
}

static int print_stats(struct threadscheduling_bpf *obj)
{
	time_t t;
	struct tm *tm;
	char ts[32];

	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", tm);

	// Clear screen for better visualization
	printf("\033[2J\033[H");

	print_header();
	printf("\n  Timestamp: %s\n", ts);

	print_sm_histogram(obj);
	print_warp_distribution(obj);
	print_thread_samples(obj);

	printf("\n  Press Ctrl+C to exit.\n");
	fflush(stdout);

	return 0;
}

int main(int argc, char **argv)
{
	struct threadscheduling_bpf *skel;
	int err;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = threadscheduling_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = threadscheduling_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}

	err = threadscheduling_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}

	printf("SM/Warp/Lane mapping probe started. Waiting for CUDA kernel executions...\n");

	while (!exiting) {
		sleep(2);
		print_stats(skel);
	}

cleanup:
	/* Clean up */
	threadscheduling_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
