#include <bpf_map/userspace/map_in_maps.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <cstdint>

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
