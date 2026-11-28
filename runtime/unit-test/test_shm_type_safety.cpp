/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf_attach_ctx.hpp"
#include "bpf_map/userspace/lpm_trie_map.hpp"
#include "bpftime_helper_group.hpp"
#include "bpftime_prog.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_shm_safe.hpp"
#include "common_def.hpp"
#include "handler/handler_manager.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include "linux/bpf.h"
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

uint64_t sample_noexcept_fn(uint64_t) noexcept { return 0; }
uint64_t sample_var_noexcept_fn(uint64_t, ...) noexcept { return 0; }
uint64_t sample_throw_fn(uint64_t) { return 0; }
uint64_t sample_var_throw_fn(uint64_t, ...) { return 0; }
template <typename T = bpftime_helper_fn>
constexpr bool helper_fn_has_public_raw_v = requires(T f) { f.raw; };
static_assert(std::is_constructible_v<bpftime_helper_fn, decltype(&sample_noexcept_fn)>);
static_assert(std::is_constructible_v<bpftime_helper_fn, decltype(&sample_var_noexcept_fn)>);
static_assert(!std::is_constructible_v<bpftime_helper_fn, decltype(&sample_throw_fn)>);
static_assert(!std::is_constructible_v<bpftime_helper_fn, decltype(&sample_var_throw_fn)>);
static_assert(!std::is_constructible_v<bpftime_helper_fn, void *>);
static_assert(!std::is_aggregate_v<bpftime_helper_fn> && !helper_fn_has_public_raw_v<>);
static_assert(std::is_nothrow_invocable_v<
	      decltype(attach::signal_callback_context::override_return),
	      uint64_t, uint64_t>);

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

TEST_CASE("Map helper catches SHM bad_alloc across JIT frame as -ENOMEM",
	  "[shm_safe][helper][noexcept]")
{
	const std::string shm_name =
		"bpftime_jit_noexcept_" + std::to_string(getpid());
	auto save_env = [](const char *k) -> std::optional<std::string> {
		const char *v = getenv(k);
		return v ? std::optional<std::string>(v) : std::nullopt;
	};
	const auto prev_name = save_env("BPFTIME_GLOBAL_SHM_NAME");
	const auto prev_mb = save_env("BPFTIME_SHM_MEMORY_MB");
	struct scope_exit {
		std::function<void()> fn;
		~scope_exit() { fn(); }
	} restore{ [&] {
		bpftime_destroy_global_shm();
		shared_memory_object::remove(shm_name.c_str());
		prev_name ? setenv("BPFTIME_GLOBAL_SHM_NAME", prev_name->c_str(), 1)
			  : unsetenv("BPFTIME_GLOBAL_SHM_NAME");
		prev_mb ? setenv("BPFTIME_SHM_MEMORY_MB", prev_mb->c_str(), 1)
			: unsetenv("BPFTIME_SHM_MEMORY_MB");
	} };

	REQUIRE(setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) == 0);
	REQUIRE(setenv("BPFTIME_SHM_MEMORY_MB", "4", 1) == 0);
	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);

	struct lpm_key4 {
		uint32_t prefixlen;
		uint32_t data;
	};
	const int map_fd = bpftime_maps_create(
		-1, "oom_lpm",
		{ .type = static_cast<uint32_t>(BPF_MAP_TYPE_LPM_TRIE),
		  .key_size = sizeof(lpm_key4),
		  .value_size = 1024 * 1024,
		  .max_ents = 16,
		  .flags = BPF_F_NO_PREALLOC });
	REQUIRE(map_fd >= 0);

	ebpf_inst insns[] = {
		{ 0x79, 2, 1, 0, 0 },	   { 0x79, 3, 1, 8, 0 },
		{ 0xb7, 1, 0, 0, map_fd }, { 0xb7, 4, 0, 0, 0 },
		{ 0x85, 0, 0, 0, 2 },	   { 0x95, 0, 0, 0, 0 },
	};
	runtime_config cfg;
	cfg.set_vm_name(DEFAULT_VM_NAME);
	bpftime_prog prog(insns, std::size(insns), "jit_oom_test",
			  std::move(cfg));
	REQUIRE(bpftime_helper_group::get_shm_maps_helper_group()
			.add_helper_group_to_prog(&prog) == 0);
	REQUIRE(prog.bpftime_prog_load(true) == 0);

	std::vector<uint8_t> big_val(1024 * 1024, 0x5a);
	bool saw_enomem = false;
	for (uint32_t i = 0; i < 8 && !saw_enomem; ++i) {
		lpm_key4 k{ 32, i + 1 };
		uint64_t args[2] = { reinterpret_cast<uint64_t>(&k),
				     reinterpret_cast<uint64_t>(big_val.data()) };
		uint64_t retval = 0;
		errno = EAGAIN;
		REQUIRE(prog.bpftime_prog_exec(args, sizeof(args), &retval) == 0);
		REQUIRE(errno == EAGAIN);
		saw_enomem = (retval == static_cast<uint64_t>(-ENOMEM));
		if (!saw_enomem)
			REQUIRE(retval == 0);
	}
	REQUIRE(saw_enomem);
}
