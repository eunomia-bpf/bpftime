/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf_attach_ctx.hpp"
#include "bpf_map/userspace/lpm_trie_map.hpp"
#include "bpftime_shm_safe.hpp"
#include "handler/handler_manager.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unistd.h>
#include <unordered_map>
#include <variant>
#include <vector>

using namespace bpftime;
using namespace boost::interprocess;

namespace
{

struct pod_shm_entry {
	uint32_t a;
	uint64_t b;
	char tag[16];
};

struct polymorphic_shm_bad {
	int x = 0;
	virtual ~polymorphic_shm_bad() = default;
};

struct pointer_member_bad {
	uint32_t len;
	void *host_addr;
};

struct ref_member_bad {
	const int &host_ref;
};

struct nested_stl_bad {
	int id;
	struct inner {
		std::vector<int> heap_vec;
	} sub;
};

static_assert(shm_safe_type<int>);
static_assert(shm_safe_type<pod_shm_entry>);
static_assert(shm_safe_type<uint64_t[8]>);
static_assert(shm_safe_type<std::optional<pod_shm_entry>>);
static_assert(shm_safe_type<std::pair<uint32_t, uint64_t>>);
static_assert(shm_safe_type<std::tuple<int, uint32_t, pod_shm_entry>>);
static_assert(shm_safe_type<std::variant<std::monostate, pod_shm_entry>>);
#ifdef BPFTIME_ENABLE_CUDA_ATTACH
static_assert(shm_safe_type<cuda::HelperCallRequest>);
static_assert(shm_safe_type<cuda::HelperCallResponse>);
static_assert(shm_safe_type<cuda::CommSharedMem>);
#endif
static_assert(shm_safe_type<handler_variant>);
static_assert(shm_safe_type<handler_manager>);

static_assert(!shm_safe_type<void *>);
static_assert(!shm_safe_type<const char *>);
static_assert(!shm_safe_type<int &>);
static_assert(!shm_safe_type<shm_init_ctx>);
static_assert(!shm_safe_type<polymorphic_shm_bad>);
static_assert(!shm_safe_type<pointer_member_bad>);
static_assert(!shm_safe_type<ref_member_bad>);
static_assert(!shm_safe_type<nested_stl_bad>);
static_assert(!shm_safe_type<std::string>);
static_assert(!shm_safe_type<std::vector<int>>);
static_assert(!shm_safe_type<std::unordered_map<int, int>>);
static_assert(!shm_safe_type<std::mutex>);
static_assert(!shm_safe_type<std::variant<int, void *>>);

struct shm_segment_cleaner {
	std::string name;
	~shm_segment_cleaner()
	{
		shared_memory_object::remove(name.c_str());
	}
};

} // namespace

TEST_CASE("shm_construct and shm_init_ctx survive creator unmap on LPM trie",
	  "[shm_safe]")
{
	const std::string shm_name =
		"bpftime_test_shm_safe_" + std::to_string(getpid());
	shared_memory_object::remove(shm_name.c_str());
	shm_segment_cleaner cleaner{ shm_name };

	struct lpm_key4 {
		uint32_t prefixlen;
		uint8_t ip[4];
	};

	{
		managed_shared_memory creator(create_only, shm_name.c_str(),
					      1 << 20);
		auto *trie = shm_construct<lpm_trie_map_impl>(
			creator, "test_lpm", shm_init_ctx(creator),
			static_cast<unsigned int>(sizeof(lpm_key4)), 8u, 64u);
		REQUIRE(trie != nullptr);

		lpm_key4 k{ 24, { 10, 0, 1, 0 } };
		uint64_t v = 0x123456789abcdef0ULL;
		REQUIRE(trie->elem_update(&k, &v, 0) == 0);
	}

	managed_shared_memory peer(open_only, shm_name.c_str());
	auto found = peer.find<lpm_trie_map_impl>("test_lpm");
	REQUIRE(found.first != nullptr);

	lpm_key4 probe{ 32, { 10, 0, 1, 42 } };
	auto *val = static_cast<uint64_t *>(found.first->elem_lookup(&probe));
	REQUIRE(val != nullptr);
	REQUIRE(*val == 0x123456789abcdef0ULL);

	lpm_key4 k2{ 16, { 10, 0, 0, 0 } };
	uint64_t v2 = 0xfeedfacecafebeefULL;
	REQUIRE(found.first->elem_update(&k2, &v2, 0) == 0);

	lpm_key4 del_k{ 24, { 10, 0, 1, 0 } };
	REQUIRE(found.first->elem_delete(&del_k) == 0);

	auto *fallback =
		static_cast<uint64_t *>(found.first->elem_lookup(&probe));
	REQUIRE(fallback != nullptr);
	REQUIRE(*fallback == 0xfeedfacecafebeefULL);

	peer.destroy_ptr(found.first);
}
