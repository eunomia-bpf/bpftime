#include "bpf_map/userspace/stack_trace_map.hpp"
#include "catch2/catch_test_macros.hpp"
#include "spdlog/cfg/env.h"
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
	std::mt19937 gen((std::random_device()()));
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
	auto stk1 = generated.begin()->second.at(0);
	int id1 = impl.fill_stack_trace(stk1, false, false);
	REQUIRE(id1 >= 0);
	// Make some collision
	auto stk2 = generated.begin()->second.at(1);

	int id2 = impl.fill_stack_trace(stk2, false, true);
	REQUIRE(id1 == id2);
	// The query result should be the same as id1
	{
		auto result = (uint64_t *)impl.elem_lookup(&id2);
		REQUIRE(std::equal(stk1.begin(), stk1.end(), result));
	}

	auto stk3 = generated.begin()->second.at(2);
	int id3 = impl.fill_stack_trace(stk3, true, true);
	// The query result should be the same as id3
	{
		auto result = (uint64_t *)impl.elem_lookup(&id3);
		REQUIRE(std::equal(stk3.begin(), stk3.end(), result));
	}
}
