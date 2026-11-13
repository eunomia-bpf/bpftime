#include "bpf_map/userspace/stack_trace_map.hpp"
#include "catch2/catch_test_macros.hpp"
#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"
#include "unit-test/common_def.hpp"
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <vector>
static const char *SHM_NAME = "BPFTIME_TEST_STACK_TRACE_MAP";
using namespace bpftime;
using namespace boost::interprocess;

TEST_CASE("Test stack trace map 1")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(create_only, SHM_NAME, 20 << 20);

	stack_trace_map_impl impl(mem, 4, 255 * 8, 10000);
	std::vector<uint64_t> test_stack{ 1, 2, 3, 4, 5 };

	int id1 = impl.fill_stack_trace(test_stack, false, false);
	REQUIRE(id1 >= 0);
	{
		uint64_t *query_result = (uint64_t *)impl.elem_lookup(&id1);
		REQUIRE(query_result != nullptr);
		for (size_t i = 0; i < test_stack.size(); i++)
			REQUIRE(query_result[i] == test_stack[i]);
	}
	REQUIRE(impl.elem_delete(&id1) == 0);
	REQUIRE((impl.elem_lookup(&id1) == nullptr && errno == ENOENT));
}
static std::map<int, std::vector<std::vector<uint64_t> > >
make_collision(int max_entries, int total_count, int vector_size)
{
	std::mt19937 gen(12345);
	std::uniform_int_distribution<uint32_t> rand(1, 1000);
	std::map<int, std::vector<std::vector<uint64_t> > > result;
	for (int i = 0; i < total_count; i++) {
		while (true) {
			std::vector<uint64_t> curr;
			for (int j = 0; j < vector_size; j++) {
				curr.push_back(rand(gen));
			}
			int hash = stack_trace_map::hash_stack_trace(curr) %
				   max_entries;
			result[hash].push_back(curr);
			if (result[hash].size() >= 5)
				break;
		}
	}
	return result;
}
TEST_CASE("Test stack trace map 2")
{
	spdlog::cfg::load_env_levels();

	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(create_only, SHM_NAME, 20 << 20);

	stack_trace_map_impl impl(mem, 4, 255 * 8, 1000);

	auto generated = make_collision(1000, 10000, 3);
	{
		bool has_collision = false;
		for (const auto &item : generated)
			if (item.second.size() > 1)
				has_collision = true;
		REQUIRE(has_collision);
	}
	auto stk1 = generated[100].at(0);
	int id1 = impl.fill_stack_trace(stk1, false, false);
	REQUIRE(id1 >= 0);
	// Make some collision
	auto stk2 = generated[100].at(1);

	int id2 = impl.fill_stack_trace(stk2, false, true);
	REQUIRE(id1 == id2);
	// The query result should be the same as id1
	{
		auto result = (uint64_t *)impl.elem_lookup(&id2);
		REQUIRE(std::equal(stk1.begin(), stk1.end(), result));
	}
	SPDLOG_INFO("Checking insertion for two same stack trace..");
	// Put two same entries into the map
	auto stk3 = generated[233].at(2);
	int id3 = impl.fill_stack_trace(stk3, true, false);

	auto stk4 = generated[233].at(2);
	int id4 = impl.fill_stack_trace(stk4, true, false);

	// The query result should be the same as id4
	{
		REQUIRE(id3 == id4);
		auto result = (uint64_t *)impl.elem_lookup(&id3);
		REQUIRE(std::equal(stk4.begin(), stk4.end(), result));
	}
}

// Search for two distinct stacks (with the given sizes) that hash into the
// same bucket. The search is deterministic through a fixed seed.
static bool
find_colliding_stacks(uint32_t max_entries, size_t size_a, size_t size_b,
		      std::vector<uint64_t> &stk_a,
		      std::vector<uint64_t> &stk_b, uint32_t seed)
{
	std::mt19937 gen(seed);
	std::uniform_int_distribution<uint64_t> rand(1, 1000);
	auto make_stack = [&](size_t n) {
		std::vector<uint64_t> v(n);
		for (auto &x : v)
			x = rand(gen);
		return v;
	};
	std::map<uint32_t, std::vector<uint64_t>> occupied;
	for (int i = 0; i < 20000; i++) {
		auto v = make_stack(size_a);
		occupied.emplace(
			stack_trace_map::hash_stack_trace(v) % max_entries, v);
	}
	for (int i = 0; i < 20000; i++) {
		auto v = make_stack(size_b);
		uint32_t bucket =
			stack_trace_map::hash_stack_trace(v) % max_entries;
		auto it = occupied.find(bucket);
		if (it != occupied.end() && it->second != v) {
			stk_a = it->second;
			stk_b = v;
			return true;
		}
	}
	return false;
}

TEST_CASE("Test stack trace map 3: equal-size different-content collision")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(create_only, SHM_NAME, 20 << 20);
	stack_trace_map_impl impl(mem, 4, 255 * 8, 1000);

	std::vector<uint64_t> stk1, stk2;
	REQUIRE(find_colliding_stacks(1000, 3, 3, stk1, stk2, 0x1234));
	REQUIRE(stk1.size() == stk2.size());
	REQUIRE(stk1 != stk2);
	REQUIRE(stack_trace_map::hash_stack_trace(stk1) % 1000 ==
		stack_trace_map::hash_stack_trace(stk2) % 1000);

	int id1 = impl.fill_stack_trace(stk1, false, false);
	REQUIRE(id1 >= 0);

	// Kernel semantics: without BPF_F_REUSE_STACKID, a collision with a
	// different stack returns -EEXIST and the original stack is kept.
	int id2 = impl.fill_stack_trace(stk2, false, false);
	REQUIRE(id2 == -EEXIST);
	{
		auto result = (uint64_t *)impl.elem_lookup(&id1);
		REQUIRE(result != nullptr);
		REQUIRE(std::equal(stk1.begin(), stk1.end(), result));
	}

	// With BPF_F_REUSE_STACKID the id is reused and the stack replaced.
	int id3 = impl.fill_stack_trace(stk2, true, false);
	REQUIRE(id3 == id1);
	{
		auto result = (uint64_t *)impl.elem_lookup(&id3);
		REQUIRE(result != nullptr);
		REQUIRE(std::equal(stk2.begin(), stk2.end(), result));
	}

	// Inserting the very same stack again deduplicates to the same id
	// instead of reporting a collision.
	int id4 = impl.fill_stack_trace(stk2, false, false);
	REQUIRE(id4 == id3);
}

TEST_CASE("Test stack trace map 4: different-size collision")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(create_only, SHM_NAME, 20 << 20);
	stack_trace_map_impl impl(mem, 4, 255 * 8, 1000);

	std::vector<uint64_t> stk1, stk2;
	REQUIRE(find_colliding_stacks(1000, 3, 5, stk1, stk2, 0x5678));
	REQUIRE(stk1.size() != stk2.size());
	REQUIRE(stack_trace_map::hash_stack_trace(stk1) % 1000 ==
		stack_trace_map::hash_stack_trace(stk2) % 1000);

	int id1 = impl.fill_stack_trace(stk1, false, false);
	REQUIRE(id1 >= 0);

	int id2 = impl.fill_stack_trace(stk2, false, false);
	REQUIRE(id2 == -EEXIST);
	{
		auto result = (uint64_t *)impl.elem_lookup(&id1);
		REQUIRE(result != nullptr);
		REQUIRE(std::equal(stk1.begin(), stk1.end(), result));
	}

	int id3 = impl.fill_stack_trace(stk2, true, false);
	REQUIRE(id3 == id1);
	{
		auto result = (uint64_t *)impl.elem_lookup(&id3);
		REQUIRE(result != nullptr);
		REQUIRE(std::equal(stk2.begin(), stk2.end(), result));
	}
}

TEST_CASE("Test stack trace map 5: fast hash comparison keeps legacy id reuse")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(create_only, SHM_NAME, 20 << 20);
	stack_trace_map_impl impl(mem, 4, 255 * 8, 1000);

	std::vector<uint64_t> stk1, stk2;
	REQUIRE(find_colliding_stacks(1000, 3, 3, stk1, stk2, 0x9abc));
	REQUIRE(stk1 != stk2);

	int id1 = impl.fill_stack_trace(stk1, false, false);
	REQUIRE(id1 >= 0);

	// BPF_F_FAST_STACK_CMP compares by hash/bucket only: distinct stacks
	// still share the id and never trigger -EEXIST.
	int id2 = impl.fill_stack_trace(stk2, false, true);
	REQUIRE(id2 == id1);
	int id3 = impl.fill_stack_trace(stk2, true, true);
	REQUIRE(id3 == id1);
}
