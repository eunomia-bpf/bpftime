// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/* GPU-local full-record device buffer collector for the kernelretsnoop
 * per-thread workload.
 *
 * The collector owns the skeleton, so the eight-bank GPU_ARRAY map (and its
 * device buffer) is created in this process at skeleton load time. A CUDA
 * context is therefore initialized here before map creation and kept alive
 * through the final collection. After the CUDA client finishes, the collector
 * performs exactly eight whole-value bpf_map_lookup_elem(bank) drains,
 * reusing one host destination allocation, and reports the bulk-drain bytes
 * and wall time separately from prefill throughput.
 *
 * This is bounded capture with post-run collection, not an unbounded
 * concurrently drained streaming ring.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "full_record_device_buffer_value.h"
#if defined(FRDB_AOSOA_LAYOUT)
#include "./.output-aosoa/full-record-device-buffer.skel.h"
#elif defined(FRDB_SOA_LAYOUT) && defined(FRDB_WARP_CONTIGUOUS_INDEX)
#include "./.output-soa-warp/full-record-device-buffer.skel.h"
#elif defined(FRDB_SOA_LAYOUT)
#include "./.output-soa/full-record-device-buffer.skel.h"
#else
#include "./.output/full-record-device-buffer.skel.h"
#endif

#define warn(...) fprintf(stderr, __VA_ARGS__)

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile sig_atomic_t exiting = 0;

static void sig_handler(int sig)
{
	exiting = true;
}

static CUcontext owner_cuda_context = NULL;

static bool init_owner_cuda_context(void)
{
	if (cuInit(0) != CUDA_SUCCESS) {
		warn("cuInit failed\n");
		return false;
	}
	CUdevice device;
	if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
		warn("cuDeviceGet failed\n");
		return false;
	}
	if (cuCtxCreate_v2(&owner_cuda_context, 0, device) != CUDA_SUCCESS ||
	    owner_cuda_context == NULL) {
		warn("cuCtxCreate failed\n");
		return false;
	}
	return true;
}

static uint64_t elapsed_ns(const struct timespec *start,
			   const struct timespec *end)
{
	return (uint64_t)(end->tv_sec - start->tv_sec) * 1000000000ULL +
	       (uint64_t)(end->tv_nsec - start->tv_nsec);
}

int main(void)
{
	/* Start the syscall server before cuInit: CUDA's first fopen otherwise
	 * re-enters the server's own cuInit while the driver is initializing. */
	uint32_t ignored_map_id = 0;
	(void)bpf_map_get_next_id(0, &ignored_map_id);
	if (!init_owner_cuda_context())
		return 1;

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	struct full_record_device_buffer_bpf *skel = NULL;
	int err = 0;
	int status = 1;

	skel = full_record_device_buffer_bpf__open();
	if (!skel) {
		warn("Failed to open and load BPF skeleton\n");
		return 1;
	}
	err = full_record_device_buffer_bpf__load(skel);
	if (err) {
		warn("Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}
	err = full_record_device_buffer_bpf__attach(skel);
	if (err) {
		warn("Failed to attach BPF skeleton: %d\n", err);
		goto cleanup;
	}

	const uint64_t value_bytes = bpf_map__value_size(skel->maps.arena);
	printf("Full-record bank count: %" PRIu64 "\n",
	       (uint64_t)FRDB_NUM_BANKS);
	printf("Full-record slots per bank: %" PRIu64 "\n",
	       (uint64_t)FRDB_SLOTS_PER_BANK);
	printf("Full-record total slots: %" PRIu64 "\n",
	       (uint64_t)FRDB_TOTAL_SLOTS);
	printf("Full-record records per slot: %" PRIu64 "\n",
	       (uint64_t)FRDB_RECORDS_PER_SLOT);
	printf("Full-record record bytes: %" PRIu64 "\n",
	       (uint64_t)sizeof(struct frdb_record));
	printf("Full-record layout: %s\n", FRDB_LAYOUT_LABEL);
	printf("Full-record value bytes (per bank): %" PRIu64 "\n", value_bytes);
	printf("Full-record probe attached; waiting for SIGINT\n");
	fflush(stdout);

	while (!exiting)
		usleep(100000);

	/* One host destination allocation, reused for every bank drain. */
	void *value = malloc(sizeof(frdb_value));
	if (!value) {
		warn("Failed to allocate whole-value drain buffer\n");
		goto cleanup;
	}

	int map_fd = bpf_map__fd(skel->maps.arena);
	uint64_t drain_time_ns = 0;
	uint64_t committed = 0, active_slots = 0;
	uint64_t overflow = 0, out_of_range = 0, nonzero_timestamps = 0;
	for (uint32_t bank = 0; bank < FRDB_NUM_BANKS; bank++) {
		struct timespec start, end;
		clock_gettime(CLOCK_MONOTONIC, &start);
		err = bpf_map_lookup_elem(map_fd, &bank, value);
		clock_gettime(CLOCK_MONOTONIC, &end);
		drain_time_ns += elapsed_ns(&start, &end);
		if (err) {
			warn("Bank %u drain lookup failed: %d (%s)\n", bank,
			     err, strerror(errno));
			free(value);
			goto cleanup;
		}
		const frdb_value *v = value;
		overflow += v->total_overflow;
		out_of_range += v->total_out_of_range;
		for (uint64_t slot = 0; slot < FRDB_SLOTS_PER_BANK; slot++) {
			const uint64_t count = v->slot_counters[slot];
			if (count == 0)
				continue;
			active_slots += 1;
			committed += count;
			for (uint64_t k = 0; k < count; k++) {
#if defined(FRDB_AOSOA_LAYOUT)
				/* The timestamp is field 9; the slot's ten
				 * fields sit at
				 * fields[k][slot / 32][f][slot % 32]. */
				nonzero_timestamps +=
					v->fields[k][slot / FRDB_GROUP_SLOTS]
						       [9]
						       [slot % FRDB_GROUP_SLOTS] != 0;
#elif defined(FRDB_SOA_LAYOUT)
				const uint64_t plane_index =
					k * FRDB_SLOTS_PER_BANK + slot;

				nonzero_timestamps +=
					v->timestamp[plane_index] != 0;
#else
				const struct frdb_record *record =
					&v->records[k *
						    FRDB_SLOTS_PER_BANK + slot];
				nonzero_timestamps += record->timestamp != 0;
#endif
			}
		}
	}
	const uint64_t drain_bytes = value_bytes * FRDB_NUM_BANKS;

	printf("Full-record drain bytes: %" PRIu64 "\n", drain_bytes);
	printf("Full-record drain time ns: %" PRIu64 "\n", drain_time_ns);
	printf("Full-record committed records: %" PRIu64 "\n", committed);
	printf("Full-record active slots: %" PRIu64 "\n", active_slots);
	printf("Full-record overflow records: %" PRIu64 "\n", overflow);
	printf("Full-record out-of-range threads: %" PRIu64 "\n", out_of_range);
	printf("Full-record nonzero timestamps: %" PRIu64 "\n",
	       nonzero_timestamps);
	fflush(stdout);
	free(value);
	status = 0;

cleanup:
	full_record_device_buffer_bpf__destroy(skel);
	if (owner_cuda_context)
		cuCtxDestroy(owner_cuda_context);
	return status;
}
