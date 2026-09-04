// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/* Copyright (c) 2020 Facebook */
#define _GNU_SOURCE
#include <errno.h>
#include <dlfcn.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <bpftime_gpu_ringbuf.h>
#include "./.output/kernelretsnoop.skel.h"
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
struct data {
	uint64_t block_x, block_y, block_z;
	uint64_t thread_x, thread_y, thread_z;
	uint64_t block_dim_x, block_dim_y, block_dim_z;
	uint64_t timestamp;
};

struct coordinate {
	uint64_t x, y, z;
};

struct state {
	uint64_t count;
	uint64_t nonzero_timestamps;
	uint64_t bad_size_records;
	uint64_t collector_errors;
	struct data *events;
	size_t events_count;
	size_t events_capacity;
};

static void poll_callback(const void *data, uint64_t size, void *ctx)
{
	struct state *state = (struct state *)ctx;
	state->count += 1;
	if (size != sizeof(struct data)) {
		state->bad_size_records += 1;
		return;
	}
	const struct data *event = data;
	state->nonzero_timestamps += event->timestamp != 0;
	if (state->events_count == state->events_capacity) {
		size_t next = state->events_capacity == 0
				      ? 4096
				      : state->events_capacity * 2;
		if (next < state->events_capacity ||
		    next > SIZE_MAX / sizeof(*state->events)) {
			state->collector_errors += 1;
			return;
		}
		void *resized = realloc(state->events,
					next * sizeof(*state->events));
		if (!resized) {
			state->collector_errors += 1;
			return;
		}
		state->events = resized;
		state->events_capacity = next;
	}
	state->events[state->events_count++] = *event;
}

static bool event_coordinate(const struct data *event,
			     struct coordinate *coordinate)
{
	const uint64_t blocks[] = {
		event->block_x, event->block_y, event->block_z,
	};
	const uint64_t threads[] = {
		event->thread_x, event->thread_y, event->thread_z,
	};
	const uint64_t dimensions[] = {
		event->block_dim_x, event->block_dim_y, event->block_dim_z,
	};
	uint64_t *values[] = {
		&coordinate->x, &coordinate->y, &coordinate->z,
	};

	for (size_t axis = 0; axis < 3; axis++) {
		if (dimensions[axis] == 0 || threads[axis] >= dimensions[axis] ||
		    blocks[axis] >
			(UINT64_MAX - threads[axis]) / dimensions[axis])
			return false;
		*values[axis] = blocks[axis] * dimensions[axis] + threads[axis];
	}
	return true;
}

static int compare_coordinates(const void *lhs_ptr, const void *rhs_ptr)
{
	const struct data *lhs = lhs_ptr;
	const struct data *rhs = rhs_ptr;
	struct coordinate lhs_coordinate = {};
	struct coordinate rhs_coordinate = {};
	const bool lhs_valid = event_coordinate(lhs, &lhs_coordinate);
	const bool rhs_valid = event_coordinate(rhs, &rhs_coordinate);
	const uint64_t lhs_values[] = {
		lhs_coordinate.x, lhs_coordinate.y, lhs_coordinate.z,
	};
	const uint64_t rhs_values[] = {
		rhs_coordinate.x, rhs_coordinate.y, rhs_coordinate.z,
	};

	/* validate_cartesian rejects invalid geometry before sorting. */
	if (lhs_valid != rhs_valid)
		return lhs_valid ? -1 : 1;
	for (size_t i = 0; i < sizeof(lhs_values) / sizeof(lhs_values[0]); i++) {
		if (lhs_values[i] < rhs_values[i])
			return -1;
		if (lhs_values[i] > rhs_values[i])
			return 1;
	}
	return 0;
}

struct cartesian_result {
	uint64_t unique_coordinates;
	uint64_t max_multiplicity;
	uint64_t multiplicity_220;
	uint64_t multiplicity_44;
	uint64_t multiplicity_22;
	uint64_t other_multiplicity;
	uint64_t segment_mismatches;
	uint64_t invalid_launch_coordinates;
	bool complete;
};

enum {
	ORACLE_LAUNCHES = 220,
	ORACLE_BLOCK_DIM_Y = 256,
	ORACLE_HIGH_BLOCKS_X = 4,
	ORACLE_MIDDLE_BLOCKS_X = 4,
	ORACLE_LOW_BLOCKS_X = 80,
	ORACLE_HIGH_COORDINATES = 1024,
	ORACLE_MIDDLE_COORDINATES = 1024,
	ORACLE_LOW_COORDINATES = 20480,
	ORACLE_UNIQUE_COORDINATES = 22528,
	ORACLE_TOTAL_EVENTS = 720896,
};
_Static_assert(ORACLE_HIGH_BLOCKS_X * ORACLE_BLOCK_DIM_Y ==
		       ORACLE_HIGH_COORDINATES &&
		       ORACLE_MIDDLE_BLOCKS_X * ORACLE_BLOCK_DIM_Y ==
		       ORACLE_MIDDLE_COORDINATES &&
		       ORACLE_LOW_BLOCKS_X * ORACLE_BLOCK_DIM_Y ==
		       ORACLE_LOW_COORDINATES,
	       "kernelretsnoop launch geometry is inconsistent");
_Static_assert(ORACLE_HIGH_COORDINATES * 220 +
		       ORACLE_MIDDLE_COORDINATES * 44 +
		       ORACLE_LOW_COORDINATES * 22 ==
		       ORACLE_TOTAL_EVENTS,
	       "kernelretsnoop multiplicity oracle total is inconsistent");

static uint64_t oracle_expected_multiplicity(
	const struct coordinate *coordinate)
{
	const uint64_t middle_end =
		ORACLE_HIGH_BLOCKS_X + ORACLE_MIDDLE_BLOCKS_X;
	const uint64_t low_end = middle_end + ORACLE_LOW_BLOCKS_X;

	/* rope_norm grows grid.x while block=(1, 256, 1). */
	if (coordinate->y >= ORACLE_BLOCK_DIM_Y || coordinate->z != 0)
		return 0;
	if (coordinate->x < ORACLE_HIGH_BLOCKS_X)
		return 220;
	if (coordinate->x < middle_end)
		return 44;
	if (coordinate->x < low_end)
		return 22;
	return 0;
}

static struct cartesian_result validate_cartesian(struct state *state)
{
	struct cartesian_result result = {};
	if (state->events_count == 0 || state->collector_errors != 0)
		return result;

	uint64_t maxima[3] = {};
	for (size_t i = 0; i < state->events_count; i++) {
		struct coordinate coordinate;

		if (!event_coordinate(&state->events[i], &coordinate)) {
			result.invalid_launch_coordinates++;
			continue;
		}
		const uint64_t values[] = {
			coordinate.x, coordinate.y, coordinate.z,
		};
		for (size_t axis = 0; axis < 3; axis++)
			if (values[axis] > maxima[axis])
				maxima[axis] = values[axis];
	}
	if (result.invalid_launch_coordinates != 0)
		return result;

	uint64_t dimensions[3];
	uint64_t expected_coordinates = 1;
	for (size_t axis = 0; axis < 3; axis++) {
		if (maxima[axis] == UINT64_MAX ||
		    expected_coordinates > UINT64_MAX / (maxima[axis] + 1))
			return result;
		dimensions[axis] = maxima[axis] + 1;
		expected_coordinates *= dimensions[axis];
	}

	qsort(state->events, state->events_count, sizeof(*state->events),
	      compare_coordinates);
	for (size_t i = 0; i < state->events_count;) {
		size_t next = i + 1;
		while (next < state->events_count &&
		       compare_coordinates(&state->events[i],
					   &state->events[next]) == 0)
			next++;
		const uint64_t multiplicity = next - i;
		if (multiplicity > result.max_multiplicity)
			result.max_multiplicity = multiplicity;
		if (multiplicity == 220)
			result.multiplicity_220++;
		else if (multiplicity == 44)
			result.multiplicity_44++;
		else if (multiplicity == 22)
			result.multiplicity_22++;
		else
			result.other_multiplicity++;

		struct coordinate coordinate;
		event_coordinate(&state->events[i], &coordinate);
		const uint64_t expected_multiplicity =
			oracle_expected_multiplicity(&coordinate);
		if (multiplicity != expected_multiplicity)
			result.segment_mismatches++;
		result.unique_coordinates++;
		i = next;
	}

	result.complete = result.unique_coordinates == expected_coordinates;
	return result;
}

static bool multiplicity_oracle_matches(const struct cartesian_result *result,
					 uint64_t total_events)
{
	return result->complete && result->max_multiplicity == ORACLE_LAUNCHES &&
	       result->multiplicity_220 == ORACLE_HIGH_COORDINATES &&
	       result->multiplicity_44 == ORACLE_MIDDLE_COORDINATES &&
	       result->multiplicity_22 == ORACLE_LOW_COORDINATES &&
	       result->other_multiplicity == 0 &&
	       result->segment_mismatches == 0 &&
	       result->invalid_launch_coordinates == 0 &&
	       result->unique_coordinates == ORACLE_UNIQUE_COORDINATES &&
	       total_events == ORACLE_TOTAL_EVENTS;
}

static void print_coordinate_validation(const struct cartesian_result *result,
					uint64_t total_events,
					int oracle_enabled,
					bool oracle_passed)
{
	printf("Cartesian launches: %" PRIu64 "\n", result->max_multiplicity);
	printf("Cartesian coordinates: %" PRIu64 "\n",
	       result->unique_coordinates);
	printf("Cartesian complete: %d\n", result->complete ? 1 : 0);
	printf("Coordinate multiplicity 220: %" PRIu64 "\n",
	       result->multiplicity_220);
	printf("Coordinate multiplicity 44: %" PRIu64 "\n",
	       result->multiplicity_44);
	printf("Coordinate multiplicity 22: %" PRIu64 "\n",
	       result->multiplicity_22);
	printf("Coordinate multiplicity other: %" PRIu64 "\n",
	       result->other_multiplicity);
	printf("Coordinate segment mismatches: %" PRIu64 "\n",
	       result->segment_mismatches);
	printf("Invalid launch coordinates: %" PRIu64 "\n",
	       result->invalid_launch_coordinates);
	printf("Unique coordinates: %" PRIu64 "\n",
	       result->unique_coordinates);
	printf("Multiplicity oracle enabled: %d\n", oracle_enabled);
	printf("Multiplicity oracle total events: %" PRIu64 "\n",
	       total_events);
	printf("Multiplicity oracle passed: %d\n",
	       oracle_passed ? 1 : 0);
}

static int multiplicity_oracle_mode(void)
{
	const char *value =
		getenv("BPFTIME_KERNELRETSNOOP_EXACT_ORACLE");
	if (!value || strcmp(value, "0") == 0)
		return 0;
	if (strcmp(value, "1") == 0)
		return 1;
	return -1;
}

static int run_multiplicity_oracle_selftest(void)
{
	struct state state = {};
	state.events = calloc(ORACLE_TOTAL_EVENTS, sizeof(*state.events));
	if (!state.events)
		return 1;
	for (uint64_t x = 0;
	     x < ORACLE_HIGH_BLOCKS_X + ORACLE_MIDDLE_BLOCKS_X +
		 ORACLE_LOW_BLOCKS_X;
	     x++) {
		const uint64_t multiplicity =
			x < ORACLE_HIGH_BLOCKS_X ? 220 :
			x < ORACLE_HIGH_BLOCKS_X + ORACLE_MIDDLE_BLOCKS_X ?
				44 : 22;
		for (uint64_t y = 0; y < ORACLE_BLOCK_DIM_Y; y++) {
			for (uint64_t occurrence = 0;
			     occurrence < multiplicity; occurrence++) {
				struct data *event =
					&state.events[state.events_count];

				event->block_x = x;
				event->thread_y = y;
				event->block_dim_x = 1;
				event->block_dim_y = ORACLE_BLOCK_DIM_Y;
				event->block_dim_z = 1;
				event->timestamp = 1;
				state.events_count++;
			}
		}
	}
	const struct cartesian_result exact = validate_cartesian(&state);
	const bool exact_passed = multiplicity_oracle_matches(
		&exact, state.events_count);
	state.events_count--;
	const struct cartesian_result missing = validate_cartesian(&state);
	const bool missing_rejected = missing.complete &&
		!multiplicity_oracle_matches(&missing, state.events_count);
	state.events_count++;
	const size_t low_segment_offset =
		ORACLE_HIGH_COORDINATES * 220 + ORACLE_MIDDLE_COORDINATES * 44;
	for (size_t i = 0; i < 220; i++)
		state.events[i].block_x =
			ORACLE_HIGH_BLOCKS_X + ORACLE_MIDDLE_BLOCKS_X;
	for (size_t i = 0; i < 22; i++) {
		struct data *event = &state.events[low_segment_offset + i];
		event->block_x = 0;
		event->thread_x = 0;
	}
	const struct cartesian_result swapped = validate_cartesian(&state);
	const bool swapped_rejected = swapped.complete &&
		swapped.multiplicity_220 == exact.multiplicity_220 &&
		swapped.multiplicity_44 == exact.multiplicity_44 &&
		swapped.multiplicity_22 == exact.multiplicity_22 &&
		!multiplicity_oracle_matches(&swapped, state.events_count);
	state.events[0].block_dim_x = 0;
	const struct cartesian_result invalid = validate_cartesian(&state);
	const bool invalid_rejected = invalid.invalid_launch_coordinates == 1 &&
		!multiplicity_oracle_matches(&invalid, state.events_count);
	print_coordinate_validation(&exact, ORACLE_TOTAL_EVENTS, 1,
				    exact_passed);
	printf("Multiplicity oracle missing-event rejected: %d\n",
	       missing_rejected ? 1 : 0);
	printf("Multiplicity oracle swapped-segment rejected: %d\n",
	       swapped_rejected ? 1 : 0);
	printf("Multiplicity oracle invalid-geometry rejected: %d\n",
	       invalid_rejected ? 1 : 0);
	free(state.events);
	return exact_passed && missing_rejected && swapped_rejected &&
		invalid_rejected ? 0 : 1;
}

static uint64_t requested_thread_slots(void)
{
	const char *value = getenv("BPFTIME_MAP_GPU_THREAD_COUNT");
	if (!value || !*value)
		return 0;
	errno = 0;
	char *end = NULL;
	const unsigned long long parsed = strtoull(value, &end, 10);
	if (errno || end == value || *end != '\0' || *value == '-')
		return 0;
	return parsed;
}

int main(int argc, char **argv)
{
	if (argc == 2 &&
	    strcmp(argv[1], "--self-test-multiplicity-oracle") == 0)
		return run_multiplicity_oracle_selftest();
	const int oracle_mode = multiplicity_oracle_mode();
	if (oracle_mode < 0) {
		fprintf(stderr,
			"BPFTIME_KERNELRETSNOOP_EXACT_ORACLE must be 0 or 1\n");
		return 2;
	}
	struct kernelretsnoop_bpf *skel = NULL;
	struct state state = {};
	int err = 0;
	int status = 0;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = kernelretsnoop_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = kernelretsnoop_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		status = 1;
		goto cleanup;
	}
	err = kernelretsnoop_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		status = 1;
		goto cleanup;
	}
	int (*poll_fn)(int, void *, void (*)(const void *, uint64_t, void *)) =
		dlsym(RTLD_DEFAULT,
		      "bpftime_syscall_server__poll_gpu_ringbuf_map");
	if (poll_fn == NULL) {
		fprintf(stderr, "GPU ring-buffer poll API is unavailable\n");
		status = 1;
		goto cleanup;
	}
	int (*stats_fn)(int, struct bpftime_gpu_ringbuf_stats *) =
		dlsym(RTLD_DEFAULT,
		      "bpftime_syscall_server__get_gpu_ringbuf_stats");
	if (stats_fn == NULL) {
		fprintf(stderr, "GPU ring-buffer stats API is unavailable\n");
		status = 1;
		goto cleanup;
	}
	int mapfd = bpf_map__fd(skel->maps.rb);
	while (!exiting) {
		err = poll_fn(mapfd, &state, poll_callback);
		if (err < 0) {
			fprintf(stderr, "Unable to poll: %d\n", err);
			status = 1;
			goto cleanup;
		}
		usleep(10000);
	}

	uint64_t final_drain = 0;
	struct bpftime_gpu_ringbuf_stats stats = {};
	for (unsigned int attempts = 0;; attempts++) {
		err = poll_fn(mapfd, &state, poll_callback);
		if (err < 0) {
			fprintf(stderr, "Unable to complete final drain: %d\n", err);
			status = 1;
			goto cleanup;
		}
		final_drain += (uint64_t)err;
		err = stats_fn(mapfd, &stats);
		if (err < 0) {
			fprintf(stderr,
				"Unable to read final GPU ring-buffer stats: %d\n",
				err);
			status = 1;
			goto cleanup;
		}
		if (stats.dirty_slots == 0 && stats.pending_records == 0)
			break;
		if (attempts == 99) {
			fprintf(stderr,
				"GPU ring buffer remained active after final drain\n");
			status = 1;
			goto cleanup;
		}
		usleep(1000);
	}
	const int second_drain = poll_fn(mapfd, &state, poll_callback);
	if (second_drain < 0) {
		fprintf(stderr, "Unable to verify empty ring buffer: %d\n",
			second_drain);
		status = 1;
		goto cleanup;
	}

	err = stats_fn(mapfd, &stats);
	if (err < 0) {
		fprintf(stderr, "Unable to read GPU ring-buffer stats: %d\n", err);
		status = 1;
		goto cleanup;
	}
	const uint64_t requested = requested_thread_slots();
	const struct cartesian_result cartesian = validate_cartesian(&state);
	const bool oracle_passed = oracle_mode == 1 &&
		multiplicity_oracle_matches(&cartesian, state.count);
	const bool oracle_gate_passed = oracle_mode == 0 || oracle_passed;

	printf("Requested thread slots: %" PRIu64 "\n", requested);
	printf("Allocated thread slots: %" PRIu64 "\n",
	       stats.allocated_thread_slots);
	printf("Ring entries per thread: %" PRIu64 "\n",
	       stats.entries_per_thread);
	printf("Record bytes: %" PRIu64 "\n", stats.value_size);
	printf("Committed events: %" PRIu64 "\n", stats.committed_records);
	printf("Total events collected: %" PRIu64 "\n", state.count);
	printf("Runtime collected events: %" PRIu64 "\n",
	       stats.collected_records);
	printf("Nonzero timestamps: %" PRIu64 "\n",
	       state.nonzero_timestamps);
	printf("OOB drops: %" PRIu64 "\n", stats.oob_drops);
	printf("Full drops: %" PRIu64 "\n", stats.full_drops);
	printf("Bad-size drops: %" PRIu64 "\n",
	       stats.bad_size_drops + state.bad_size_records);
	printf("Other drops: %" PRIu64 "\n",
	       stats.other_drops + state.collector_errors);
	printf("Dirty slots: %" PRIu64 "\n", stats.dirty_slots);
	printf("Pending events: %" PRIu64 "\n", stats.pending_records);
	printf("Final drain events: %" PRIu64 "\n", final_drain);
	printf("Second drain events: %d\n", second_drain);
	print_coordinate_validation(&cartesian, state.count, oracle_mode,
				    oracle_passed);

	const bool passed =
		requested != 0 && requested == stats.allocated_thread_slots &&
		stats.entries_per_thread >= 256 &&
		stats.value_size == sizeof(struct data) &&
		stats.committed_records == state.count &&
		stats.collected_records == state.count &&
		state.nonzero_timestamps == state.count &&
		stats.oob_drops == 0 && stats.full_drops == 0 &&
		stats.bad_size_drops == 0 && state.bad_size_records == 0 &&
		stats.other_drops == 0 && state.collector_errors == 0 &&
		stats.dirty_slots == 0 && stats.pending_records == 0 &&
		second_drain == 0 && cartesian.complete && oracle_gate_passed;
	printf("Collector gate passed: %d\n", passed ? 1 : 0);
	fflush(stdout);
	if (!passed)
		status = 1;
cleanup:
	/* Clean up */
	free(state.events);
	kernelretsnoop_bpf__destroy(skel);

	return status;
}
