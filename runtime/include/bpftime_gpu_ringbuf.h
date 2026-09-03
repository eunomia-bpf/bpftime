/* SPDX-License-Identifier: MIT */
#ifndef BPFTIME_GPU_RINGBUF_H
#define BPFTIME_GPU_RINGBUF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct bpftime_gpu_ringbuf_stats {
	uint64_t value_size;
	uint64_t entries_per_thread;
	uint64_t allocated_thread_slots;
	uint64_t committed_records;
	uint64_t collected_records;
	uint64_t pending_records;
	uint64_t oob_drops;
	uint64_t full_drops;
	uint64_t bad_size_drops;
	uint64_t other_drops;
	uint64_t dirty_slots;
};

#ifdef __cplusplus
}
#endif

#endif /* BPFTIME_GPU_RINGBUF_H */
