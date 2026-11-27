#include <bpf_map/userspace/map_in_maps.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <thread>
#include <vector>

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

TEST_CASE("Hash-of-maps supports lock-free concurrent helper lookups",
	  "[maps][map_in_map][concurrent]")
{
	REQUIRE_FALSE(hash_map_of_maps_impl::should_lock);

	static const char *shm_name = "bpftime_hash_of_maps_concurrent_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 1024 * 1024);
	hash_map_of_maps_impl map(segment, sizeof(uint32_t), 128, 0);

	constexpr uint32_t steady_keys = 16;
	for (uint32_t k = 0; k < steady_keys; ++k) {
		int fd = static_cast<int>(100 + k);
		REQUIRE(map.elem_update(&k, &fd, 0) == 0);
	}

	std::atomic<bool> stop{ false };
	std::atomic<uint64_t> failures{ 0 };
	std::thread writer([&]() {
		uint32_t churn_base = 64;
		for (int iter = 0; iter < 500; ++iter) {
			uint32_t k = churn_base + static_cast<uint32_t>(iter % 32);
			int fd = 1000 + (iter % 100);
			if (map.elem_update(&k, &fd, 0) != 0)
				failures.fetch_add(1);
			if (iter % 2 == 1 && map.elem_delete(&k) != 0)
				failures.fetch_add(1);
		}
		stop.store(true, std::memory_order_release);
	});

	std::vector<std::thread> readers;
	for (int t = 0; t < 8; ++t) {
		readers.emplace_back([&]() {
			while (!stop.load(std::memory_order_acquire)) {
				for (uint32_t k = 0; k < steady_keys; ++k) {
					auto val = reinterpret_cast<uintptr_t>(
						map.elem_lookup(&k));
					if (val != 100 + k)
						failures.fetch_add(1);
				}
			}
		});
	}

	writer.join();
	for (auto &r : readers)
		r.join();
	REQUIRE(failures.load() == 0);
}
