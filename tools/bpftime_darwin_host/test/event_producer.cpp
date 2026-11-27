// SPDX-License-Identifier: MIT
//
// Test-only in-process event producer. Opens the daemon's SHM segment with
// SHM_OPEN_ONLY (the same path a DYLD_INSERT agent uses) and emits ringbuf
// or software-perf records through the public runtime APIs, so the native
// e2e test can exercise subscribe_events without a preload agent.
//
// Usage:
//   bpftime_darwin_event_producer <shm_name> ring <ring_fd> <count>
//   bpftime_darwin_event_producer <shm_name> perf <perf_fd> <count>
#include "bpftime_shm.hpp"
#include <string>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace
{
constexpr size_t kPayloadSize = 32;

void fill_payload(uint8_t *payload, uint64_t seq)
{
	memset(payload, 0, kPayloadSize);
	memcpy(payload, &seq, sizeof(seq));
}

int produce_ring(int fd, int count)
{
	for (int i = 0; i < count; i++) {
		void *slot = nullptr;
		for (int retry = 0; retry < 1000; retry++) {
			slot = bpftime_ringbuf_reserve(fd, kPayloadSize);
			if (slot != nullptr)
				break;
			usleep(1000);
		}
		if (slot == nullptr)
			return 1;
		fill_payload(static_cast<uint8_t *>(slot), (uint64_t)i);
		bpftime_ringbuf_submit(fd, slot, 0);
	}
	return 0;
}

int produce_perf(int fd, int count)
{
	if (bpftime_get_software_perf_event_raw_buffer(
		    fd, (1 + 64) * static_cast<size_t>(getpagesize())) ==
	    nullptr)
		return 1;
	uint8_t payload[kPayloadSize];
	for (int i = 0; i < count; i++) {
		fill_payload(payload, (uint64_t)i);
		if (bpftime_perf_event_output(fd, payload, kPayloadSize) != 0)
			return 1;
	}
	return 0;
}
} // namespace

int main(int argc, char **argv)
{
	if (argc != 5) {
		fprintf(stderr, "usage: %s <shm_name> ring|perf <fd> <count>\n",
			argv[0]);
		return 2;
	}
	setenv("BPFTIME_GLOBAL_SHM_NAME", argv[1], 1);
	bpftime_initialize_global_shm(bpftime::shm_open_type::SHM_OPEN_ONLY);
	int fd = atoi(argv[3]);
	int count = atoi(argv[4]);
	if (count <= 0)
		return 2;
	return std::string(argv[2]) == "ring" ? produce_ring(fd, count) :
	       std::string(argv[2]) == "perf" ? produce_perf(fd, count) :
						2;
}
