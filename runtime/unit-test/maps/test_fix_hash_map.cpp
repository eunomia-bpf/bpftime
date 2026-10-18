#include <bpf_map/bpftime_hash_map.hpp>
#include <bpf_map/userspace/fix_hash_map.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <catch2/catch_test_macros.hpp>
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
