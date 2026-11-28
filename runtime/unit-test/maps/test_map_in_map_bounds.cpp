#include <bpf_map/userspace/map_in_maps.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <thread>

using namespace boost::interprocess;
using namespace bpftime;

namespace
{
struct shm_cleanup {
	explicit shm_cleanup(const char *name) : name(name)
	{
		shared_memory_object::remove(name);
	}
	~shm_cleanup()
	{
		shared_memory_object::remove(name);
	}
	const char *name;
};
} // namespace

TEST_CASE("Array-of-maps rejects out-of-range lookup keys",
	  "[maps][map_in_map]")
{
	static const char *shm_name = "bpftime_map_in_map_bounds_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 65536);
	array_map_of_maps_impl map(segment, 2);

	uint32_t invalid_key = 2;
	errno = 0;
	REQUIRE(map.elem_lookup(&invalid_key) == nullptr);
	REQUIRE(errno == ENOENT);
}

TEST_CASE("Hash-of-maps stores inner map descriptors",
	  "[maps][map_in_map]")
{
	static const char *shm_name = "bpftime_hash_of_maps_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 65536);
	hash_map_of_maps_impl map(segment, sizeof(uint32_t), 2, 0);

	uint32_t key = 42;
	int inner_fd = 17;
	REQUIRE(map.elem_update(&key, &inner_fd, 0) == 0);
	REQUIRE(map.elem_lookup(&key) == reinterpret_cast<void *>(17));

	uint32_t next_key = 0;
	REQUIRE(map.map_get_next_key(nullptr, &next_key) == 0);
	REQUIRE(next_key == key);
	REQUIRE(map.elem_delete(&key) == 0);
	REQUIRE(map.elem_lookup(&key) == nullptr);
	REQUIRE(errno == ENOENT);
}

TEST_CASE("Array-of-maps syscall lookup returns null for empty slot",
	  "[maps][map_in_map]")
{
	static const char *shm_name = "bpftime_array_map_in_map_syscall_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 65536);
	array_map_of_maps_impl map(segment, 2);

	uint32_t key = 0;
	// Never-updated slot stores 0; the syscall path returns nullptr (the
	// syscall layer maps that to ENOENT) instead of a pointer to the zero
	// value that memcpy would copy as a bogus inner fd.
	REQUIRE(map.elem_lookup_userspace(&key) == nullptr);

	// Out-of-range key also returns nullptr through the userspace path.
	uint32_t invalid_key = 2;
	REQUIRE(map.elem_lookup_userspace(&invalid_key) == nullptr);

	int inner_fd = 17;
	REQUIRE(map.elem_update(&key, &inner_fd, 0) == 0);
	auto *stored =
		static_cast<int *>(map.elem_lookup_userspace(&key));
	REQUIRE(stored != nullptr);
	REQUIRE(*stored == 17);
}

TEST_CASE("Hash-of-maps syscall lookup rejects zero inner fd",
	  "[maps][map_in_map]")
{
	static const char *shm_name = "bpftime_hash_map_in_map_syscall_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 65536);
	hash_map_of_maps_impl map(segment, sizeof(uint32_t), 4, 0);

	uint32_t key = 7;
	// Missing key: underlying lookup returns nullptr.
	REQUIRE(map.elem_lookup_userspace(&key) == nullptr);

	// Explicitly stored zero must also be rejected (exercises the *stored==0
	// branch, distinct from the nullptr branch above).
	int zero_fd = 0;
	REQUIRE(map.elem_update(&key, &zero_fd, 0) == 0);
	REQUIRE(map.elem_lookup_userspace(&key) == nullptr);

	int inner_fd = 17;
	REQUIRE(map.elem_update(&key, &inner_fd, 0) == 0);
	auto *stored =
		static_cast<int *>(map.elem_lookup_userspace(&key));
	REQUIRE(stored != nullptr);
	REQUIRE(*stored == 17);
}

TEST_CASE("Hash-of-maps lookups never see a half-written inner id",
	  "[maps][map_in_map]")
{
	static const char *shm_name = "bpftime_hash_of_maps_torn_id_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 65536);
	hash_map_of_maps_impl map(segment, sizeof(uint32_t), 4, 0);

	// The ids differ in every byte, so a lookup racing a byte-wise copy
	// would see a mix of the two.
	const int ids[2] = { 0x01020304, 0x7f6e5d4c };
	uint32_t key = 1;
	REQUIRE(map.elem_update(&key, &ids[0], 0) == 0);
	std::atomic<bool> stop{ false };
	std::atomic<uint64_t> torn{ 0 }, failed{ 0 };
	std::thread reader([&]() {
		while (!stop.load(std::memory_order_relaxed)) {
			const int id = (int)(uintptr_t)map.elem_lookup(&key);
			if (id != ids[0] && id != ids[1])
				torn.fetch_add(1);
		}
	});
	for (int i = 0; i < 200000; ++i)
		if (map.elem_update(&key, &ids[i & 1], 0) != 0)
			failed.fetch_add(1);
	stop.store(true);
	reader.join();
	REQUIRE(failed.load() == 0);
	REQUIRE(torn.load() == 0);
}

TEST_CASE("Hash-of-maps lookups never return the id of a key that reused the slot",
	  "[maps][map_in_map]")
{
	static const char *shm_name = "bpftime_hash_of_maps_slot_reuse_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 65536);
	hash_map_of_maps_impl map(segment, sizeof(uint32_t), 4, 0);

	// a and b take turns in one slot: the free list hands back the most
	// recently freed one, so b lands where a was.
	const uint32_t a = 1, b = 2;
	const int id_a = 0x0a0a0a0a, id_b = 0x0b0b0b0b;
	REQUIRE(map.elem_update(&a, &id_a, 0) == 0);
	std::atomic<bool> stop{ false };
	std::atomic<uint64_t> hits{ 0 }, wrong{ 0 };
	std::thread reader([&]() {
		while (!stop.load(std::memory_order_relaxed)) {
			const int id = (int)(uintptr_t)map.elem_lookup(&a);
			if (id == id_a)
				hits.fetch_add(1, std::memory_order_relaxed);
			else if (id != 0)
				wrong.fetch_add(1, std::memory_order_relaxed);
		}
	});
	for (int i = 0; i < 200000; ++i) {
		(void)map.elem_delete(&a);
		(void)map.elem_update(&b, &id_b, 0);
		(void)map.elem_delete(&b);
		(void)map.elem_update(&a, &id_a, 0);
	}
	stop.store(true);
	reader.join();
	REQUIRE(hits.load() > 0);
	REQUIRE(wrong.load() == 0);
}

TEST_CASE("Hash-of-maps syscall lookups never return the id of a key that reused the slot",
	  "[maps][map_in_map]")
{
	static const char *shm_name =
		"bpftime_hash_of_maps_syscall_slot_reuse_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 65536);
	hash_map_of_maps_impl map(segment, sizeof(uint32_t), 4, 0);

	// Same race through the syscall wrapper, which hands back a thread-local
	// snapshot: a lookup of `a` must return `a`'s id or nothing, never the
	// id of `b` that took over the freed slot.
	const uint32_t a = 1, b = 2;
	const int id_a = 0x0a0a0a0a, id_b = 0x0b0b0b0b;
	REQUIRE(map.elem_update(&a, &id_a, 0) == 0);
	std::atomic<bool> stop{ false };
	std::atomic<uint64_t> hits{ 0 }, wrong{ 0 };
	std::thread reader([&]() {
		while (!stop.load(std::memory_order_relaxed)) {
			auto *id = static_cast<int *>(
				map.elem_lookup_userspace(&a));
			if (id == nullptr)
				continue;
			if (*id == id_a)
				hits.fetch_add(1, std::memory_order_relaxed);
			else
				wrong.fetch_add(1, std::memory_order_relaxed);
		}
	});
	for (int i = 0; i < 200000; ++i) {
		(void)map.elem_delete(&a);
		(void)map.elem_update(&b, &id_b, 0);
		(void)map.elem_delete(&b);
		(void)map.elem_update(&a, &id_a, 0);
	}
	stop.store(true);
	reader.join();
	REQUIRE(hits.load() > 0);
	REQUIRE(wrong.load() == 0);
}
