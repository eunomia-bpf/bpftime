#include "catch2/catch_test_macros.hpp"

#include "handler/perf_event_handler.hpp"
#include "common_def.hpp"
#include <atomic>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace
{
struct event_payload {
	int producer;
	int sequence;
};

constexpr size_t aligned_perf_record_size(size_t payload_size)
{
	constexpr size_t alignment = sizeof(uint64_t);
	return (sizeof(bpftime::perf_sample_raw) + payload_size + alignment -
		1) &
	       ~(alignment - 1);
}

void copy_from_perf_ring(uint8_t *base, size_t ring_size, uint64_t offset,
			 void *dst, size_t size)
{
	uint8_t *copy_start_1 = base + (offset & (ring_size - 1));
	if (size + copy_start_1 <= base + ring_size) {
		memcpy(dst, copy_start_1, size);
	} else {
		size_t len_first = base + ring_size - copy_start_1;
		size_t len_second = size - len_first;
		memcpy(dst, copy_start_1, len_first);
		memcpy((uint8_t *)dst + len_first, base, len_second);
	}
}
} // namespace

TEST_CASE("Software perf event buffers shard concurrent producers by thread",
	  "[perf_event][software_perf_event]")
{
	const std::string shared_memory_name =
		"SoftwarePerfEventShardTestShm-" + std::to_string(getpid());
	const size_t shared_memory_size = 16 * 1024 * 1024;
	shm_remove remover{ std::string(shared_memory_name) };

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, shared_memory_name.c_str(),
		shared_memory_size);

	auto *perf = shm.construct<bpftime::software_perf_event_data>(
		"perf")(0, 0, 0, shm);
	REQUIRE(perf != nullptr);

	const size_t ring_size = 1024 * 1024;
	void *raw_buffer = perf->ensure_mmap_buffer(getpagesize() + ring_size);
	REQUIRE(raw_buffer != nullptr);

	constexpr int producer_count = 4;
	constexpr int events_per_producer = 256;
	std::atomic<bool> start{ false };
	std::atomic<bool> output_failed{ false };
	std::vector<std::thread> producers;
	producers.reserve(producer_count);
	for (int producer = 0; producer < producer_count; producer++) {
		producers.emplace_back([&, producer]() {
			while (!start.load(std::memory_order_acquire)) {
				std::this_thread::yield();
			}
			for (int sequence = 0; sequence < events_per_producer;
			     sequence++) {
				event_payload payload{ producer, sequence };
				if (perf->output_data(&payload,
						      sizeof(payload)) != 0) {
					output_failed.store(
						true,
						std::memory_order_release);
				}
			}
		});
	}

	start.store(true, std::memory_order_release);
	for (auto &producer : producers) {
		producer.join();
	}
	REQUIRE_FALSE(output_failed.load(std::memory_order_acquire));

	REQUIRE(perf->has_data());
	// Shards of exited threads are only reclaimed once every
	// `software_perf_event_reclaim_drain_interval` (64) drains, and a joined
	// thread can still answer `tgkill` until the kernel reaps it, so retry a
	// few reclamation rounds instead of relying on a single one.
	bool shards_reclaimed = false;
	for (int round = 0; round < 32 && !shards_reclaimed; round++) {
		for (int i = 0; i < 64; i++) {
			REQUIRE(perf->has_data());
		}
		shards_reclaimed = perf->producer_shards.empty();
		if (!shards_reclaimed) {
			std::this_thread::sleep_for(
				std::chrono::milliseconds(10));
		}
	}
	REQUIRE(shards_reclaimed);

	auto *header = (perf_event_mmap_page *)raw_buffer;
	auto *base = (uint8_t *)raw_buffer + getpagesize();
	uint64_t tail = header->data_tail;
	uint64_t head = header->data_head;
	REQUIRE(head > tail);

	std::vector<int> seen(producer_count * events_per_producer, 0);
	int record_count = 0;
	while (tail < head) {
		bpftime::perf_sample_raw sample;
		copy_from_perf_ring(base, ring_size, tail, &sample,
				    sizeof(sample));
		REQUIRE(sample.header.type == PERF_RECORD_SAMPLE);
		REQUIRE(sample.header.size ==
			aligned_perf_record_size(sizeof(event_payload)));
		REQUIRE(sample.size ==
			sample.header.size - sizeof(bpftime::perf_sample_raw));

		event_payload payload;
		copy_from_perf_ring(base, ring_size,
				    tail + sizeof(bpftime::perf_sample_raw),
				    &payload, sizeof(payload));
		REQUIRE(payload.producer >= 0);
		REQUIRE(payload.producer < producer_count);
		REQUIRE(payload.sequence >= 0);
		REQUIRE(payload.sequence < events_per_producer);
		seen[payload.producer * events_per_producer +
		     payload.sequence]++;
		record_count++;
		tail += sample.header.size;
	}

	REQUIRE(record_count == producer_count * events_per_producer);
	for (int count : seen) {
		REQUIRE(count == 1);
	}

	constexpr size_t max_payload_size =
		std::numeric_limits<uint16_t>::max() -
		sizeof(bpftime::perf_sample_raw) - (sizeof(uint64_t) - 1);
	std::vector<uint8_t> boundary_payload(max_payload_size + 1);
	const uint64_t head_before_boundary = header->data_head;
	REQUIRE(perf->output_data(boundary_payload.data(), max_payload_size) ==
		0);
	REQUIRE(perf->has_data());
	REQUIRE(header->data_head - head_before_boundary ==
		aligned_perf_record_size(max_payload_size));
	errno = 0;
	REQUIRE(perf->output_data(boundary_payload.data(),
				  boundary_payload.size()) == -1);
	REQUIRE(errno == E2BIG);
}

TEST_CASE("Software perf event records wrap on aligned boundaries after resize",
	  "[perf_event][software_perf_event]")
{
	const std::string shared_memory_name =
		"SoftwarePerfEventResizeTestShm-" + std::to_string(getpid());
	const size_t shared_memory_size = 16 * 1024 * 1024;
	shm_remove remover{ std::string(shared_memory_name) };

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, shared_memory_name.c_str(),
		shared_memory_size);

	auto *perf = shm.construct<bpftime::software_perf_event_data>(
		"perf")(0, 0, 0, shm);
	REQUIRE(perf != nullptr);

	event_payload dropped_before_mmap{ 0, 0 };
	REQUIRE(perf->output_data(&dropped_before_mmap,
				  sizeof(dropped_before_mmap)) == 0);
	REQUIRE_FALSE(perf->has_data());

	const size_t ring_size = 64;
	void *raw_buffer = perf->ensure_mmap_buffer(getpagesize() + ring_size);
	REQUIRE(raw_buffer != nullptr);

	auto *header = (perf_event_mmap_page *)raw_buffer;
	auto *base = (uint8_t *)raw_buffer + getpagesize();
	// Without 8-byte record alignment, the fourth 20-byte record would
	// start at offset 60 and split its 8-byte perf_event_header.
	for (int sequence = 0; sequence < 4; sequence++) {
		event_payload payload{ 1, sequence };
		REQUIRE(perf->output_data(&payload, sizeof(payload)) == 0);
		REQUIRE(perf->has_data());

		const uint64_t tail = header->data_tail;
		const uint64_t head = header->data_head;
		REQUIRE((tail & (ring_size - 1)) + sizeof(perf_event_header) <=
			ring_size);

		perf_event_header record_header;
		copy_from_perf_ring(base, ring_size, tail, &record_header,
				    sizeof(record_header));
		REQUIRE(record_header.type == PERF_RECORD_SAMPLE);
		REQUIRE(record_header.size ==
			aligned_perf_record_size(sizeof(event_payload)));

		event_payload actual;
		copy_from_perf_ring(base, ring_size,
				    tail + sizeof(bpftime::perf_sample_raw),
				    &actual, sizeof(actual));
		REQUIRE(actual.producer == payload.producer);
		REQUIRE(actual.sequence == payload.sequence);
		REQUIRE(tail + record_header.size == head);
		header->data_tail = head;
	}
}

TEST_CASE("Software perf event mmap reports shared memory exhaustion",
	  "[perf_event][software_perf_event]")
{
	const std::string shared_memory_name =
		"SoftwarePerfEventExhaustionTestShm-" +
		std::to_string(getpid());
	const size_t shared_memory_size = 1024 * 1024;
	shm_remove remover{ std::string(shared_memory_name) };

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, shared_memory_name.c_str(),
		shared_memory_size);

	auto *perf = shm.construct<bpftime::software_perf_event_data>(
		"perf")(0, 0, 0, shm);
	REQUIRE(perf != nullptr);

	const size_t ring_size = 2 * 1024 * 1024;
	errno = 0;
	REQUIRE(perf->ensure_mmap_buffer(getpagesize() + ring_size) == nullptr);
	REQUIRE(errno == ENOMEM);
}
