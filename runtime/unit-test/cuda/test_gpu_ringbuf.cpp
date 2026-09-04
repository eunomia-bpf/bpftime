#include "bpf_map/gpu/nv_gpu_ringbuf_map.hpp"
#include "catch2/catch_test_macros.hpp"
#include <array>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace bpftime
{
struct nv_gpu_ringbuf_test_access {
	static unsigned char *buffer(nv_gpu_ringbuf_map_impl &map)
	{
		return map.data_buffer.data();
	}
	static uint64_t entry_size(const nv_gpu_ringbuf_map_impl &map)
	{
		return map.entry_size;
	}
	static uint64_t record_stride(const nv_gpu_ringbuf_map_impl &map)
	{
		return map.record_stride;
	}
};
} // namespace bpftime

namespace
{
using namespace bpftime;
using namespace boost::interprocess;

static_assert(alignof(ringbuf_header) == alignof(uint64_t));
static_assert(offsetof(ringbuf_header, head) == 0);
static_assert(offsetof(ringbuf_header, tail) == 8);
static_assert(offsetof(ringbuf_header, dirty) == 16);
static_assert(sizeof(ringbuf_header) == 24);

class shared_memory_fixture {
    public:
	explicit shared_memory_fixture(const char *suffix)
		: name_("bpftime_gpu_ringbuf_" + std::to_string(getpid()) +
			"_" + suffix),
		  memory_(create_only, name_.c_str(), 1 << 20)
	{
	}

	~shared_memory_fixture()
	{
		shared_memory_object::remove(name_.c_str());
	}

	managed_shared_memory &memory()
	{
		return memory_;
	}

    private:
	std::string name_;
	managed_shared_memory memory_;
};

void append(nv_gpu_ringbuf_map_impl &map, uint64_t thread, uint64_t value,
	    uint64_t capacity)
{
	auto *base = nv_gpu_ringbuf_test_access::buffer(map);
	const uint64_t entry_size =
		nv_gpu_ringbuf_test_access::entry_size(map);
	auto *header = reinterpret_cast<ringbuf_header *>(base +
							 thread * entry_size);
	const uint64_t slot = header->tail % capacity;
	auto *record = reinterpret_cast<unsigned char *>(header) +
		       sizeof(*header) +
		       slot * nv_gpu_ringbuf_test_access::record_stride(map);
	*reinterpret_cast<uint64_t *>(record) = sizeof(value);
	std::memcpy(record + sizeof(uint64_t), &value, sizeof(value));
	header->tail++;
}
} // namespace

TEST_CASE("GPU ring buffer drains every pending record", "[gpu-ringbuf]")
{
	shared_memory_fixture fixture("drain");
	nv_gpu_ringbuf_map_impl map(fixture.memory(), sizeof(uint64_t), 3, 2);
	append(map, 0, 10, 3);
	append(map, 0, 11, 3);
	append(map, 0, 12, 3);
	append(map, 1, 20, 3);
	append(map, 1, 21, 3);

	bpftime_gpu_ringbuf_stats before = {};
	REQUIRE(map.get_stats(&before) == 0);
	REQUIRE(before.committed_records == 5);
	REQUIRE(before.pending_records == 5);

	std::vector<uint64_t> values;
	const int drained = map.drain_data([&](const void *data, uint64_t size) {
		REQUIRE(size == sizeof(uint64_t));
		values.push_back(*static_cast<const uint64_t *>(data));
	});
	REQUIRE(drained == 5);
	REQUIRE(values == std::vector<uint64_t>{ 10, 11, 12, 20, 21 });
	REQUIRE(map.drain_data([](const void *, uint64_t) {}) == 0);

	bpftime_gpu_ringbuf_stats after = {};
	REQUIRE(map.get_stats(&after) == 0);
	REQUIRE(after.committed_records == 5);
	REQUIRE(after.collected_records == 5);
	REQUIRE(after.pending_records == 0);
	REQUIRE(map.get_stats(nullptr) == -EINVAL);
}

TEST_CASE("GPU ring buffer reports dirty and producer failures",
	  "[gpu-ringbuf]")
{
	shared_memory_fixture fixture("stats");
	nv_gpu_ringbuf_map_impl map(fixture.memory(), sizeof(uint64_t), 2, 2);
	append(map, 0, 42, 2);
	append(map, 1, 84, 2);
	auto *base = nv_gpu_ringbuf_test_access::buffer(map);
	auto *header = reinterpret_cast<ringbuf_header *>(base);
	header->dirty = 1;

	const uint64_t entry_size =
		nv_gpu_ringbuf_test_access::entry_size(map);
	auto *errors = reinterpret_cast<ringbuf_error_counters *>(base +
							     entry_size * 2);
	errors->oob_drops = 1;
	errors->full_drops = 2;
	errors->bad_size_drops = 3;
	errors->other_drops = 4;

	REQUIRE(map.drain_data([](const void *, uint64_t) {}) == 1);
	bpftime_gpu_ringbuf_stats stats = {};
	REQUIRE(map.get_stats(&stats) == 0);
	REQUIRE(stats.dirty_slots == 1);
	REQUIRE(stats.pending_records == 1);
	REQUIRE(stats.oob_drops == 1);
	REQUIRE(stats.full_drops == 2);
	REQUIRE(stats.bad_size_drops == 3);
	REQUIRE(stats.other_drops == 4);

	header->dirty = 0;
	REQUIRE(map.drain_data([](const void *, uint64_t) {}) == 1);
	REQUIRE(map.get_stats(&stats) == 0);
	REQUIRE(stats.dirty_slots == 0);
	REQUIRE(stats.pending_records == 0);
}

TEST_CASE("GPU ring buffer aligns arbitrary value sizes", "[gpu-ringbuf]")
{
	shared_memory_fixture fixture("alignment");
	nv_gpu_ringbuf_map_impl map(fixture.memory(), 12, 2, 2);
	auto *base = nv_gpu_ringbuf_test_access::buffer(map);
	const uint64_t entry_size =
		nv_gpu_ringbuf_test_access::entry_size(map);
	const uint64_t stride =
		nv_gpu_ringbuf_test_access::record_stride(map);
	REQUIRE(entry_size % alignof(uint64_t) == 0);
	REQUIRE(stride % alignof(uint64_t) == 0);
	REQUIRE(reinterpret_cast<uintptr_t>(base + entry_size) %
			alignof(uint64_t) ==
		0);

	auto *header = reinterpret_cast<ringbuf_header *>(base);
	auto *record = base + sizeof(*header);
	const std::array<unsigned char, 12> expected = {
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
	};
	*reinterpret_cast<uint64_t *>(record) = expected.size();
	std::memcpy(record + sizeof(uint64_t), expected.data(), expected.size());
	header->tail = 1;
	REQUIRE(map.drain_data([&](const void *data, uint64_t size) {
		REQUIRE(size == expected.size());
		REQUIRE(std::memcmp(data, expected.data(), expected.size()) == 0);
	}) == 1);
}

TEST_CASE("GPU ring buffer validates degenerate sizes", "[gpu-ringbuf]")
{
	shared_memory_fixture fixture("sizes");
	REQUIRE_THROWS_AS((nv_gpu_ringbuf_map_impl(
				 fixture.memory(), UINT64_MAX, 1, 1)),
			  std::overflow_error);

	nv_gpu_ringbuf_map_impl empty(fixture.memory(), sizeof(uint64_t), 0, 1);
	REQUIRE(empty.drain_data([](const void *, uint64_t) {}) == -EINVAL);

	nv_gpu_ringbuf_map_impl zero(fixture.memory(), sizeof(uint64_t), 1, 1);
	auto *base = nv_gpu_ringbuf_test_access::buffer(zero);
	auto *header = reinterpret_cast<ringbuf_header *>(base);
	*reinterpret_cast<uint64_t *>(base + sizeof(*header)) = 0;
	header->tail = 1;
	bool observed = false;
	REQUIRE(zero.drain_data([&](const void *, uint64_t size) {
		observed = true;
		REQUIRE(size == 0);
	}) == 1);
	REQUIRE(observed);
}

TEST_CASE("GPU ring buffer rejects corrupt pending records", "[gpu-ringbuf]")
{
	shared_memory_fixture fixture("corrupt");
	nv_gpu_ringbuf_map_impl map(fixture.memory(), sizeof(uint64_t), 2, 1);
	auto *base = nv_gpu_ringbuf_test_access::buffer(map);
	auto *header = reinterpret_cast<ringbuf_header *>(base);
	header->tail = 3;
	REQUIRE(map.drain_data([](const void *, uint64_t) {}) == -EOVERFLOW);

	header->tail = 1;
	auto *record = base + sizeof(*header);
	*reinterpret_cast<uint64_t *>(record) = sizeof(uint64_t) + 1;
	REQUIRE(map.drain_data([](const void *, uint64_t) {}) == -EMSGSIZE);
	REQUIRE(header->head == 0);

	header->head = 2;
	header->tail = 1;
	bpftime_gpu_ringbuf_stats stats = {};
	REQUIRE(map.get_stats(&stats) == 0);
	REQUIRE(stats.other_drops == 1);
	REQUIRE(map.drain_data([](const void *, uint64_t) {}) == -EOVERFLOW);
}

TEST_CASE("GPU ring buffer reports unsupported map operations", "[gpu-ringbuf]")
{
	shared_memory_fixture fixture("unsupported");
	nv_gpu_ringbuf_map_impl map(fixture.memory(), sizeof(uint64_t), 1, 1);

	errno = 0;
	REQUIRE(map.elem_lookup(nullptr) == nullptr);
	REQUIRE(errno == ENOTSUP);

	errno = 0;
	REQUIRE(map.elem_update(nullptr, nullptr, 0) == -1);
	REQUIRE(errno == ENOTSUP);

	errno = 0;
	REQUIRE(map.elem_delete(nullptr) == -1);
	REQUIRE(errno == ENOTSUP);

	errno = 0;
	REQUIRE(map.map_get_next_key(nullptr, nullptr) == -1);
	REQUIRE(errno == ENOTSUP);
}
