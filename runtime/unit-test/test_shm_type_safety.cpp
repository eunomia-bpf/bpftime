/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf_attach_ctx.hpp"
#ifdef BPFTIME_BUILD_WITH_LIBBPF
#include "bpf_map/shared/array_map_kernel_user.hpp"
#endif
#include "bpf_map/userspace/lpm_trie_map.hpp"
#include "bpftime_shm_safe.hpp"
#include "handler/handler_manager.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <optional>
#include <string>
#include <sys/wait.h>
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
static_assert(shm_safe_type<shm_local_key>);
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

struct dummy_local_fd {
	int fd;
	pid_t owner_pid;
};

shm_local_table<dummy_local_fd, 64> g_test_local_table{};

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

TEST_CASE("shm_local_table negative caching, eviction, and fork isolation",
	  "[shm_safe]")
{
	auto close_dummy = [](dummy_local_fd &e) noexcept {
		if (e.fd >= 0) {
			close(e.fd);
			e.fd = -1;
		}
	};

	int pfd[2] = { -1, -1 };
	REQUIRE(pipe(pfd) == 0);

	shm_local_key ok_key{ 3u, 11u };
	int init_calls = 0;
	errno = EAGAIN;
	auto *entry = g_test_local_table.get_or_init(
		ok_key,
		[&](dummy_local_fd &out) {
			++init_calls;
			errno = EINVAL;
			out.fd = pfd[0];
			out.owner_pid = getpid();
			return true;
		},
		close_dummy);
	REQUIRE(entry != nullptr);
	REQUIRE(entry->fd == pfd[0]);
	REQUIRE(init_calls == 1);
	REQUIRE(errno == EAGAIN);

	auto *hit = g_test_local_table.get_or_init(
		ok_key, [&](dummy_local_fd &) {
			++init_calls;
			return true;
		});
	REQUIRE(hit == entry);
	REQUIRE(init_calls == 1);

	// Load-time pthread_atfork bumps epoch across fork(): child must not
	// reuse the parent's cached entry.
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		int child_inits = 0;
		auto *c_entry = g_test_local_table.get_or_init(
			ok_key, [&](dummy_local_fd &out) {
				++child_inits;
				out.fd = 99;
				out.owner_pid = getpid();
				return true;
			});
		const bool ok = (c_entry != nullptr && child_inits == 1 &&
				 c_entry->fd == 99 &&
				 c_entry->owner_pid == getpid());
		_exit(ok ? 0 : 1);
	}
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
	REQUIRE(fcntl(pfd[0], F_GETFD) >= 0);

	// Stale generation in the same process is cleaned up and taken over.
	int pfd2[2] = { -1, -1 };
	REQUIRE(pipe(pfd2) == 0);
	shm_local_key next_gen_key{ 3u, 12u };
	auto *replaced = g_test_local_table.get_or_init(
		next_gen_key,
		[&](dummy_local_fd &out) {
			out.fd = pfd2[0];
			out.owner_pid = getpid();
			return true;
		},
		close_dummy);
	REQUIRE(replaced != nullptr);
	REQUIRE(replaced->fd == pfd2[0]);
	errno = 0;
	REQUIRE(fcntl(pfd[0], F_GETFD) == -1);
	REQUIRE(errno == EBADF);
	close(pfd[1]);

	g_test_local_table.evict(next_gen_key, close_dummy);
	errno = 0;
	REQUIRE(fcntl(pfd2[0], F_GETFD) == -1);
	REQUIRE(errno == EBADF);
	REQUIRE(g_test_local_table.find(next_gen_key) == nullptr);
	close(pfd2[1]);

	// Negative cache: failed open runs once and replays errno on later hits.
	shm_local_key bad_key{ 7u, 19u };
	int fail_calls = 0;
	errno = 0;
	REQUIRE(g_test_local_table.get_or_init(
			bad_key, [&](dummy_local_fd &) {
				++fail_calls;
				errno = ENOENT;
				return false;
			}) == nullptr);
	REQUIRE(fail_calls == 1);
	REQUIRE(errno == ENOENT);

	errno = EAGAIN;
	REQUIRE(g_test_local_table.get_or_init(
			bad_key, [&](dummy_local_fd &) {
				++fail_calls;
				return true;
			}) == nullptr);
	REQUIRE(fail_calls == 1);
	REQUIRE(errno == ENOENT);
	g_test_local_table.evict(bad_key, [](dummy_local_fd &) {});
}

#if defined(__linux__) && defined(BPFTIME_BUILD_WITH_LIBBPF)
TEST_CASE("array_map_kernel_user_impl isolates local state outside SHM",
	  "[shm_safe]")
{
	const std::string shm_name =
		"bpftime_test_ku_array_" + std::to_string(getpid());
	shared_memory_object::remove(shm_name.c_str());
	shm_segment_cleaner cleaner{ shm_name };

	{
		managed_shared_memory creator(create_only, shm_name.c_str(),
					      1 << 20);
		auto *arr = shm_construct<array_map_kernel_user_impl>(
			creator, "ku_nonkernel", shm_init_ctx(creator), -1, 8u,
			4u);
		uint32_t k = 2;
		uint64_t v = 0xdeadbeefcafebabeULL;
		REQUIRE(arr->elem_update(&k, &v, 0) == 0);

		auto *invalid = shm_construct<array_map_kernel_user_impl>(
			creator, "ku_invalid", shm_init_ctx(creator),
			0x7fffffff, 8u, 4u);
		REQUIRE(invalid->elem_lookup(&k) == nullptr);
	}

	managed_shared_memory peer(open_only, shm_name.c_str());
	auto found = peer.find<array_map_kernel_user_impl>("ku_nonkernel");
	REQUIRE(found.first != nullptr);
	uint32_t k = 2;
	auto *got = static_cast<uint64_t *>(found.first->elem_lookup(&k));
	REQUIRE(got != nullptr);
	REQUIRE(*got == 0xdeadbeefcafebabeULL);

	auto inv = peer.find<array_map_kernel_user_impl>("ku_invalid");
	REQUIRE(inv.first != nullptr);
	errno = 0;
	REQUIRE(inv.first->elem_lookup(&k) == nullptr);
	const int first_err = errno;
	errno = 0;
	REQUIRE(inv.first->elem_lookup(&k) == nullptr);
	REQUIRE(errno == first_err);

	peer.destroy_ptr(inv.first);
	peer.destroy_ptr(found.first);
}
#endif
