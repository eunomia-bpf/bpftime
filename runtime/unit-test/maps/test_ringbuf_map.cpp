#include <bpf_map/userspace/ringbuf_map.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <string>
#include <unistd.h>

using namespace bpftime;
using namespace boost::interprocess;

TEST_CASE("Ringbuf submit resolves a sample that wraps past the last header",
	  "[maps][ringbuf]")
{
	const std::string shm_name =
		"bpftime_ringbuf_wrap_test_" + std::to_string(getpid());
	shared_memory_object::remove(shm_name.c_str());
	struct cleanup {
		const std::string &name;
		~cleanup()
		{
			shared_memory_object::remove(name.c_str());
		}
	} cleanup{ shm_name };
	managed_shared_memory memory(create_only, shm_name.c_str(), 65536);
	constexpr uint32_t ring_size = 64;
	ringbuf_map_impl map(ring_size, memory);

	auto *consumer = static_cast<unsigned long *>(map.get_consumer_page());
	auto *producer = static_cast<unsigned long *>(map.get_producer_page());
	*consumer = ring_size - 8;
	*producer = ring_size - 8;
	auto *data = reinterpret_cast<uint8_t *>(producer) + getpagesize();

	uint64_t payload = 0x1122334455667788ULL;
	void *sample = map.reserve(sizeof(payload), 17);
	REQUIRE(sample == data + ring_size);
	REQUIRE(static_cast<int32_t *>(sample)[-1] == 17);
	memcpy(sample, &payload, sizeof(payload));
	map.submit(sample, false);
	REQUIRE(memcmp(data, &payload, sizeof(payload)) == 0);

	uint64_t observed = 0;
	auto callback = [](void *ctx, void *value, size_t size) {
		REQUIRE(size == sizeof(uint64_t));
		memcpy(ctx, value, size);
		return 0;
	};
	REQUIRE(map.create_impl_shared_ptr()->fetch_data(callback, &observed) ==
		1);
	REQUIRE(observed == payload);
}
