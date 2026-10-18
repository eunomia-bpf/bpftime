#include "catch2/catch_test_macros.hpp"

#include "handler/perf_event_handler.hpp"
#include "bpftime_shm.hpp"
#include "common_def.hpp"
#include <atomic>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <cerrno>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#if __linux__
#include <linux/bpf.h>
#include <sched.h>

extern "C" uint64_t bpf_perf_event_output(uint64_t, uint64_t, uint64_t,
					  uint64_t, uint64_t);

TEST_CASE("Perf output preserves the calling thread's CPU affinity",
	  "[perf_event][affinity]")
{
	cpu_set_t before;
	REQUIRE(sched_getaffinity(0, sizeof(before), &before) == 0);
	if (CPU_COUNT(&before) < 2)
		SKIP("Affinity restoration requires at least two allowed CPUs");

	const char *old_name = getenv("BPFTIME_GLOBAL_SHM_NAME");
	struct cleanup {
		cpu_set_t affinity;
		bool had_name;
		std::string old_name;
		int map_fd = -1;
		int perf_fd = -1;
		~cleanup()
		{
			sched_setaffinity(0, sizeof(affinity), &affinity);
			for (int fd : { map_fd, perf_fd }) {
				if (fd >= 0) {
					bpftime_close(fd);
					close(fd);
				}
			}
			bpftime_destroy_global_shm();
			bpftime_initialize_global_shm(
				bpftime::shm_open_type::SHM_NO_CREATE);
			if (had_name)
				setenv("BPFTIME_GLOBAL_SHM_NAME",
				       old_name.c_str(), 1);
			else
				unsetenv("BPFTIME_GLOBAL_SHM_NAME");
		}
	} cleanup{ before, old_name != nullptr, old_name ? old_name : "" };
	const std::string name =
		"PerfOutputAffinityTestShm-" + std::to_string(getpid());
	shm_remove remover{ std::string(name) };
	REQUIRE(setenv("BPFTIME_GLOBAL_SHM_NAME", name.c_str(), 1) == 0);
	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);

	int &map_fd = cleanup.map_fd;
	uint64_t expected_ret = static_cast<uint64_t>(-1);
	int expected_errno = ENOENT;
	SECTION("Unknown map")
	{
	}
	SECTION("CPU index outside the perf array")
	{
		// A one-entry map has no slot for any CPU other than CPU 0.
		CPU_CLR(0, &before);
		if (CPU_COUNT(&before) < 2)
			SKIP("This case requires two allowed nonzero CPUs");
		REQUIRE(sched_setaffinity(0, sizeof(before), &before) == 0);
		map_fd = bpftime_maps_create(
			-1, "short_perf_array",
			{ .type = BPF_MAP_TYPE_PERF_EVENT_ARRAY,
			  .key_size = 4,
			  .value_size = 4,
			  .max_ents = 1 });
		REQUIRE(map_fd >= 0);
		expected_errno = EINVAL;
	}
	SECTION("Successful output")
	{
		map_fd = bpftime_maps_create(
			-1, "perf_array",
			{ .type = BPF_MAP_TYPE_PERF_EVENT_ARRAY,
			  .key_size = 4,
			  .value_size = 4,
			  .max_ents = CPU_SETSIZE });
		REQUIRE(map_fd >= 0);
		int &perf_fd = cleanup.perf_fd;
		perf_fd = bpftime_add_software_perf_event(0, 0, 0);
		REQUIRE(perf_fd >= 0);
		REQUIRE(bpftime_get_software_perf_event_raw_buffer(
				perf_fd, 2 * getpagesize()) != nullptr);
		for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
			if (CPU_ISSET(cpu, &before))
				REQUIRE(bpftime_map_update_elem(map_fd, &cpu,
								&perf_fd,
								0) == 0);
		}
		expected_ret = 0;
		expected_errno = 0;
	}

	uint32_t payload = 42;
	errno = 0;
	const auto ret =
		bpf_perf_event_output(0, map_fd, 0,
				      reinterpret_cast<uint64_t>(&payload),
				      sizeof(payload));
	const int output_errno = errno;
	cpu_set_t after;
	REQUIRE(sched_getaffinity(0, sizeof(after), &after) == 0);
	CHECK(ret == expected_ret);
	if (expected_ret == static_cast<uint64_t>(-1))
		CHECK(output_errno == expected_errno);
	CHECK(CPU_EQUAL(&before, &after));
}
#endif

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
