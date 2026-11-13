#include <bpf_map/bpftime_hash_map.hpp>
#include <bpf_map/userspace/fix_hash_map.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cerrno>
#include <linux/bpf.h>

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

TEST_CASE("Fixed hash iteration scans adjusted bucket count",
	  "[maps][hash][iteration]")
{
	static const char *shm_name = "bpftime_fix_hash_iteration_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 65536);

	constexpr size_t requested_buckets = 10;
	const size_t actual_buckets =
		bpftime_hasher::next_prime(requested_buckets);
	REQUIRE(actual_buckets > requested_buckets);

	fix_size_hash_map_impl map(segment, requested_buckets, sizeof(uint32_t),
				   sizeof(uint64_t));

	uint32_t key = 0;
	for (;; ++key) {
		if (bpftime_hasher::hash_func(&key, sizeof(key)) % actual_buckets ==
		    actual_buckets - 1)
			break;
	}
	uint64_t value = 0x123456789abcdef0ULL;
	REQUIRE(map.elem_update(&key, &value, 0) == 0);

	uint32_t next_key = 0;
	REQUIRE(map.map_get_next_key(nullptr, &next_key) == 0);
	REQUIRE(next_key == key);
}

TEST_CASE("Fixed hash map propagates update flags and failures",
	  "[maps][hash]")
{
	static const char *shm_name = "bpftime_fix_hash_errors_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 65536);

	fix_size_hash_map_impl map(segment, 1, sizeof(uint32_t),
				   sizeof(uint64_t));
	uint32_t key = 1, other_key = 2, missing_key = 99;
	uint64_t v1 = 11, v2 = 22;

	REQUIRE(map.elem_update(&key, &v1, BPF_ANY) == 0);

	// BPF_NOEXIST on an existing key fails and keeps the old value
	errno = 0;
	REQUIRE(map.elem_update(&key, &v2, BPF_NOEXIST) == -1);
	REQUIRE(errno == EEXIST);
	REQUIRE(*static_cast<uint64_t *>(map.elem_lookup(&key)) == v1);

	// BPF_EXIST on a missing key fails
	errno = 0;
	REQUIRE(map.elem_update(&other_key, &v1, BPF_EXIST) == -1);
	REQUIRE(errno == ENOENT);

	// Unknown flags are rejected
	errno = 0;
	REQUIRE(map.elem_update(&other_key, &v1, 3) == -1);
	REQUIRE(errno == EINVAL);

	// The single entry fills the capacity; new inserts must be reported
	errno = 0;
	REQUIRE(map.elem_update(&other_key, &v1, BPF_ANY) == -1);
	REQUIRE(errno == E2BIG);
	REQUIRE(map.elem_lookup(&other_key) == nullptr);

	// Updating the existing key still succeeds while the map is full
	REQUIRE(map.elem_update(&key, &v2, BPF_EXIST) == 0);
	REQUIRE(*static_cast<uint64_t *>(map.elem_lookup(&key)) == v2);

	// Deleting a missing key fails; deleting an existing key works
	errno = 0;
	REQUIRE(map.elem_delete(&missing_key) == -1);
	REQUIRE(errno == ENOENT);
	REQUIRE(map.elem_delete(&key) == 0);
	REQUIRE(map.elem_lookup(&key) == nullptr);
}

TEST_CASE("Fixed hash map supports inserts, overwrites and reinsertion",
	  "[maps][hash]")
{
	static const char *shm_name = "bpftime_fix_hash_success_test";
	shm_cleanup cleanup(shm_name);
	managed_shared_memory segment(create_only, shm_name, 65536);

	fix_size_hash_map_impl map(segment, 2, sizeof(uint32_t),
				   sizeof(uint64_t));
	uint32_t key = 1, other_key = 2, missing_key = 99;
	uint64_t v1 = 11, v2 = 22, v3 = 33;

	// Deleting from an empty map reports ENOENT
	errno = 0;
	REQUIRE(map.elem_delete(&missing_key) == -1);
	REQUIRE(errno == ENOENT);

	// BPF_NOEXIST successfully inserts a missing key
	errno = 0;
	REQUIRE(map.elem_update(&key, &v1, BPF_NOEXIST) == 0);
	REQUIRE(errno == 0);
	REQUIRE(*static_cast<uint64_t *>(map.elem_lookup(&key)) == v1);

	// BPF_ANY overwrites the existing value
	errno = 0;
	REQUIRE(map.elem_update(&key, &v2, BPF_ANY) == 0);
	REQUIRE(errno == 0);
	REQUIRE(*static_cast<uint64_t *>(map.elem_lookup(&key)) == v2);

	// BPF_ANY inserts a second key within capacity
	errno = 0;
	REQUIRE(map.elem_update(&other_key, &v3, BPF_ANY) == 0);
	REQUIRE(errno == 0);
	REQUIRE(*static_cast<uint64_t *>(map.elem_lookup(&other_key)) == v3);

	// A deleted key can be reinserted, freeing capacity
	REQUIRE(map.elem_delete(&key) == 0);
	REQUIRE(map.elem_lookup(&key) == nullptr);
	errno = 0;
	REQUIRE(map.elem_update(&key, &v1, BPF_NOEXIST) == 0);
	REQUIRE(errno == 0);
	REQUIRE(*static_cast<uint64_t *>(map.elem_lookup(&key)) == v1);

	// Unknown flags, including bits above the low 32, are rejected and
	// leave the stored value untouched
	errno = 0;
	REQUIRE(map.elem_update(&key, &v2, 1ULL << 32) == -1);
	REQUIRE(errno == EINVAL);
	REQUIRE(*static_cast<uint64_t *>(map.elem_lookup(&key)) == v1);
}
