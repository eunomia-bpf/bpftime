#include "catch2/catch_test_macros.hpp"

#include "handler/perf_event_handler.hpp"
#include "common_def.hpp"
#include <atomic>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <cerrno>
#include <cstring>
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
	for (int i = 0; i < 64; i++) {
		REQUIRE(perf->has_data());
	}
	REQUIRE(perf->producer_shards.empty());

	auto *header = (perf_event_mmap_page *)raw_buffer;
	auto *base = (uint8_t *)raw_buffer + getpagesize();
	uint64_t tail = header->data_tail;
	uint64_t head = header->data_head;
	REQUIRE(head > tail);

	std::vector<int> seen(producer_count * events_per_producer, 0);
	int record_count = 0;
	while (tail < head) {
		perf_event_header record_header;
		copy_from_perf_ring(base, ring_size, tail, &record_header,
				    sizeof(record_header));
		REQUIRE(record_header.type == PERF_RECORD_SAMPLE);
		REQUIRE(record_header.size == sizeof(bpftime::perf_sample_raw) +
						      sizeof(event_payload));

		std::vector<uint8_t> record(record_header.size);
		copy_from_perf_ring(base, ring_size, tail, record.data(),
				    record.size());
		auto *sample = (const bpftime::perf_sample_raw *)record.data();
		REQUIRE(sample->size == sizeof(event_payload));

		event_payload payload;
		memcpy(&payload,
		       record.data() + sizeof(bpftime::perf_sample_raw),
		       sizeof(payload));
		REQUIRE(payload.producer >= 0);
		REQUIRE(payload.producer < producer_count);
		REQUIRE(payload.sequence >= 0);
		REQUIRE(payload.sequence < events_per_producer);
		seen[payload.producer * events_per_producer +
		     payload.sequence]++;
		record_count++;
		tail += record_header.size;
	}

	REQUIRE(record_count == producer_count * events_per_producer);
	for (int count : seen) {
		REQUIRE(count == 1);
	}
}

TEST_CASE("Software perf event producer shards rotate after mmap resize",
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

	const size_t ring_size = 64 * 1024;
	void *raw_buffer = perf->ensure_mmap_buffer(getpagesize() + ring_size);
	REQUIRE(raw_buffer != nullptr);

	event_payload payload{ 1, 7 };
	REQUIRE(perf->output_data(&payload, sizeof(payload)) == 0);
	REQUIRE(perf->has_data());

	auto *header = (perf_event_mmap_page *)raw_buffer;
	auto *base = (uint8_t *)raw_buffer + getpagesize();
	uint64_t tail = header->data_tail;
	uint64_t head = header->data_head;
	REQUIRE(head > tail);

	perf_event_header record_header;
	copy_from_perf_ring(base, ring_size, tail, &record_header,
			    sizeof(record_header));
	REQUIRE(record_header.type == PERF_RECORD_SAMPLE);
	REQUIRE(record_header.size ==
		sizeof(bpftime::perf_sample_raw) + sizeof(event_payload));

	std::vector<uint8_t> record(record_header.size);
	copy_from_perf_ring(base, ring_size, tail, record.data(),
			    record.size());
	auto *sample = (const bpftime::perf_sample_raw *)record.data();
	REQUIRE(sample->size == sizeof(event_payload));

	event_payload actual;
	memcpy(&actual, record.data() + sizeof(bpftime::perf_sample_raw),
	       sizeof(actual));
	REQUIRE(actual.producer == payload.producer);
	REQUIRE(actual.sequence == payload.sequence);
	REQUIRE(tail + record_header.size == head);
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
