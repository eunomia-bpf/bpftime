#include "catch2/catch_test_macros.hpp"
#include "linux/bpf.h"
#include "unit-test/common_def.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include "bpf_map/userspace/var_hash_map.hpp"
static const char *SHM_NAME = "_HASH_MAP_TEST";

using namespace boost::interprocess;
using namespace bpftime;

TEST_CASE("Test hash map (kernel)")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(boost::interprocess::create_only, SHM_NAME,
				  20 << 20);

	long long key, next_key, first_key, value;
	var_size_hash_map_impl map(mem, sizeof(key), sizeof(value), 2);
	key = 1;
	value = 1234;
	/* Insert key=1 element. */
	REQUIRE(map.elem_update(&key, &value, BPF_ANY) == 0);

	value = 0;
	/* BPF_NOEXIST means add new element if it doesn't exist. */
	REQUIRE(map.elem_update(&key, &value, BPF_NOEXIST) < 0);
	REQUIRE(/* key=1 already exists. */ errno == EEXIST);

	/* -1 is an invalid flag. */
	REQUIRE(map.elem_update(&key, &value, -1) < 0);
	REQUIRE(errno == EINVAL);

	/* Check that key=1 can be found. */
	{
		auto ptr = map.elem_lookup(&key);
		REQUIRE(ptr != nullptr);
		REQUIRE(*(long long *)ptr == 1234);
	}

	key = 2;
	value = 1234;
	/* Insert key=2 element. */
	REQUIRE(map.elem_update(&key, &value, BPF_ANY) == 0);

	/* Check that key=2 matches the value and delete it */
	REQUIRE(map.lookup_and_delete(&key, &value) == 0);
	REQUIRE(value == 1234);

	/* Check that key=2 is not found. */
	REQUIRE(map.elem_lookup(&key) == nullptr);
	REQUIRE(errno == ENOENT);

	/* BPF_EXIST means update existing element. */
	REQUIRE(map.elem_update(&key, &value, BPF_EXIST) < 0);

	REQUIRE(/* key=2 is not there. */
		errno == ENOENT);

	/* Insert key=2 element. */
	REQUIRE(map.elem_update(&key, &value, BPF_NOEXIST) == 0);

	/* key=1 and key=2 were inserted, check that key=0 cannot be
	 * inserted due to max_entries limit.
	 */
	key = 0;
	REQUIRE(map.elem_update(&key, &value, BPF_NOEXIST) < 0);
	REQUIRE(errno == E2BIG);

	/* Update existing element, though the map is full. */
	key = 1;
	REQUIRE(map.elem_update(&key, &value, BPF_EXIST) == 0);
	key = 2;
	REQUIRE(map.elem_update(&key, &value, BPF_ANY) == 0);
	key = 3;
	REQUIRE(map.elem_update(&key, &value, BPF_NOEXIST) < 0);
	REQUIRE(errno == E2BIG);

	/* Check that key = 0 doesn't exist. */
	key = 0;
	REQUIRE(map.elem_delete(&key) < 0);
	REQUIRE(errno == ENOENT);
	/* Iterate over two elements. */
	REQUIRE(map.map_get_next_key(NULL, &first_key) == 0);
	REQUIRE((first_key == 1 || first_key == 2));
	REQUIRE(map.map_get_next_key(&key, &next_key) == 0);
	REQUIRE((next_key == first_key));
	REQUIRE(map.map_get_next_key(&next_key, &next_key) == 0);
	REQUIRE(((next_key == 1 || next_key == 2) && (next_key != first_key)));
	REQUIRE(map.map_get_next_key(&next_key, &next_key) < 0);
	REQUIRE(errno == ENOENT);

	/* Delete both elements. */
	key = 1;
	REQUIRE(map.elem_delete(&key) == 0);
	key = 2;
	REQUIRE(map.elem_delete(&key) == 0);
	REQUIRE(map.elem_delete(&key) < 0);
	REQUIRE(errno == ENOENT);

	key = 0;
	/* Check that map is empty. */
	REQUIRE(map.map_get_next_key(NULL, &next_key) < 0);
	REQUIRE(errno == ENOENT);
	REQUIRE(map.map_get_next_key(&key, &next_key) < 0);
	REQUIRE(errno == ENOENT);
}
