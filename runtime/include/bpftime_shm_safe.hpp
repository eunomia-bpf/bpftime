/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_SHM_SAFE_HPP
#define _BPFTIME_SHM_SAFE_HPP

#include <atomic>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace bpftime
{

class shm_init_ctx;

// Compact POD slot + generation handle stored inside SHM objects instead of
// process-local OS file descriptors, mmap addresses, or driver handles.
struct shm_local_key {
	uint32_t slot = 0;
	uint32_t gen = 0;
};

inline constexpr std::size_t DEFAULT_SHM_LOCAL_TABLE_CAPACITY = 8192;

struct shm_noop_cleanup {
	template <typename T>
	constexpr void operator()(T &) const noexcept
	{
	}
};

namespace detail
{

extern std::atomic<uint32_t> g_shm_local_fork_epoch;

struct shm_local_key_counter {
	std::atomic<uint32_t> next_gen{ 1 };
	std::atomic<uint64_t>
		live_slots[DEFAULT_SHM_LOCAL_TABLE_CAPACITY / 64]{};
};

template <typename T> struct is_forbidden_shm_std_type : std::false_type {};

template <typename CharT, typename Traits, typename Alloc>
struct is_forbidden_shm_std_type<std::basic_string<CharT, Traits, Alloc>>
	: std::true_type {};

template <typename T, typename Alloc>
struct is_forbidden_shm_std_type<std::vector<T, Alloc>> : std::true_type {};

template <typename K, typename V, typename Cmp, typename Alloc>
struct is_forbidden_shm_std_type<std::map<K, V, Cmp, Alloc>>
	: std::true_type {};

template <typename K, typename V, typename Cmp, typename Alloc>
struct is_forbidden_shm_std_type<std::multimap<K, V, Cmp, Alloc>>
	: std::true_type {};

template <typename K, typename V, typename Hash, typename Eq, typename Alloc>
struct is_forbidden_shm_std_type<std::unordered_map<K, V, Hash, Eq, Alloc>>
	: std::true_type {};

template <typename K, typename V, typename Hash, typename Eq, typename Alloc>
struct is_forbidden_shm_std_type<std::unordered_multimap<K, V, Hash, Eq, Alloc>>
	: std::true_type {};

template <typename K, typename Cmp, typename Alloc>
struct is_forbidden_shm_std_type<std::set<K, Cmp, Alloc>> : std::true_type {};

template <typename K, typename Hash, typename Eq, typename Alloc>
struct is_forbidden_shm_std_type<std::unordered_set<K, Hash, Eq, Alloc>>
	: std::true_type {};

template <typename Sig>
struct is_forbidden_shm_std_type<std::function<Sig>> : std::true_type {};

template <typename T, typename Deleter>
struct is_forbidden_shm_std_type<std::unique_ptr<T, Deleter>>
	: std::true_type {};

template <typename T>
struct is_forbidden_shm_std_type<std::shared_ptr<T>> : std::true_type {};

template <typename T>
struct is_forbidden_shm_std_type<std::weak_ptr<T>> : std::true_type {};

template <> struct is_forbidden_shm_std_type<std::mutex> : std::true_type {};
template <>
struct is_forbidden_shm_std_type<std::recursive_mutex> : std::true_type {};

template <typename T> struct is_shm_safe_impl;

struct shm_any_field {
	template <typename U> operator U() const noexcept;
};

struct shm_safe_field {
	template <typename U>
	requires(is_shm_safe_impl<U>::value)
	operator U() const noexcept;
};

template <typename T, std::size_t... I>
constexpr bool can_brace_any(std::index_sequence<I...>) noexcept
{
	return requires { T{ (static_cast<void>(I), shm_any_field{})... }; };
}

template <typename T, std::size_t... I>
constexpr bool can_brace_safe(std::index_sequence<I...>) noexcept
{
	return requires { T{ (static_cast<void>(I), shm_safe_field{})... }; };
}

template <typename T, std::size_t N = 8>
constexpr bool small_aggregate_shm_safe() noexcept
{
	if constexpr (N == 0) {
		return true;
	} else if constexpr (can_brace_any<T>(std::make_index_sequence<N>{})) {
		return can_brace_safe<T>(std::make_index_sequence<N>{});
	} else {
		return small_aggregate_shm_safe<T, N - 1>();
	}
}

template <typename T> constexpr bool aggregate_shm_safe() noexcept
{
	if constexpr (!std::is_aggregate_v<T> || std::is_union_v<T> ||
		      !std::is_copy_constructible_v<T>) {
		return true;
	} else if constexpr (!std::is_default_constructible_v<T>) {
		return false;
	} else if constexpr (can_brace_any<T>(std::make_index_sequence<9>{})) {
		return true;
	} else {
		return small_aggregate_shm_safe<T>();
	}
}

template <typename T> struct is_shm_safe_compound : std::true_type {};

template <typename E>
struct is_shm_safe_compound<std::optional<E>> : is_shm_safe_impl<E> {};

template <typename A, typename B>
struct is_shm_safe_compound<std::pair<A, B>>
	: std::bool_constant<is_shm_safe_impl<A>::value &&
			     is_shm_safe_impl<B>::value> {};

template <typename... Es>
struct is_shm_safe_compound<std::tuple<Es...>>
	: std::bool_constant<(is_shm_safe_impl<Es>::value && ...)> {};

template <typename... Es>
struct is_shm_safe_compound<std::variant<Es...>>
	: std::bool_constant<(is_shm_safe_impl<Es>::value && ...)> {};

template <typename E, typename Alloc>
struct is_shm_safe_compound<boost::interprocess::vector<E, Alloc>>
	: is_shm_safe_impl<E> {};

template <typename T> struct is_shm_safe_unqualified {
	static constexpr bool value =
		!std::is_pointer_v<T> && !std::is_member_pointer_v<T> &&
		!std::is_reference_v<T> && !std::is_polymorphic_v<T> &&
		!std::is_same_v<T, shm_init_ctx> &&
		!is_forbidden_shm_std_type<T>::value &&
		is_shm_safe_compound<T>::value && aggregate_shm_safe<T>();
};

template <typename E, std::size_t N> struct is_shm_safe_unqualified<E[N]> {
	static constexpr bool value = is_shm_safe_impl<E>::value;
};

template <typename T> struct is_shm_safe_impl {
	static constexpr bool value =
		!std::is_pointer_v<T> && !std::is_reference_v<T> &&
		is_shm_safe_unqualified<std::remove_cv_t<T>>::value;
};

} // namespace detail

template <typename T>
concept shm_safe_type = detail::is_shm_safe_impl<T>::value;

template <shm_safe_type T, typename Memory, typename NameOrTag,
	  typename... Args>
T *shm_construct(Memory &memory, const NameOrTag &name_or_tag, Args &&...args)
{
	return memory.template construct<T>(name_or_tag)(
		std::forward<Args>(args)...);
}

template <shm_safe_type T, typename Memory, typename... Args>
T *shm_find_or_construct(Memory &memory, const char *name, Args &&...args)
{
	return memory.template find_or_construct<T>(name)(
		std::forward<Args>(args)...);
}

class shm_init_ctx {
    public:
	using segment_manager_t =
		boost::interprocess::managed_shared_memory::segment_manager;

	shm_init_ctx(boost::interprocess::managed_shared_memory &mem) noexcept
		: sm_(mem.get_segment_manager())
	{
	}
	explicit shm_init_ctx(segment_manager_t *sm) noexcept : sm_(sm)
	{
	}

	shm_init_ctx(const shm_init_ctx &) = delete;
	shm_init_ctx &operator=(const shm_init_ctx &) = delete;
	shm_init_ctx(shm_init_ctx &&) = delete;
	shm_init_ctx &operator=(shm_init_ctx &&) = delete;

	segment_manager_t *segment_manager() const noexcept
	{
		return sm_;
	}

	template <typename T>
	boost::interprocess::allocator<T, segment_manager_t>
	allocator() const noexcept
	{
		return boost::interprocess::allocator<T, segment_manager_t>(
			sm_);
	}

	shm_local_key allocate_local_key() const
	{
		auto *ctr = shm_find_or_construct<detail::shm_local_key_counter>(
			*sm_, "bpftime_shm_local_keys");
		for (uint32_t w = 0;
		     w < DEFAULT_SHM_LOCAL_TABLE_CAPACITY / 64; ++w) {
			uint64_t cur = ctr->live_slots[w].load(
				std::memory_order_relaxed);
			while (~cur != 0ULL) {
				const uint32_t bit = static_cast<uint32_t>(
					__builtin_ctzll(~cur));
				if (ctr->live_slots[w].compare_exchange_weak(
					    cur, cur | (1ULL << bit),
					    std::memory_order_acq_rel,
					    std::memory_order_relaxed)) {
					uint32_t gen =
						ctr->next_gen.fetch_add(
							1u,
							std::memory_order_relaxed) &
						0x3fffffffU;
					return shm_local_key{
						w * 64u + bit,
						gen == 0u ? 1u : gen,
					};
				}
			}
		}
		return shm_local_key{ 0u, 0u };
	}

    private:
	segment_manager_t *sm_;
};

inline void release_local_key(
	boost::interprocess::managed_shared_memory::segment_manager *sm,
	shm_local_key key) noexcept
{
	if (sm == nullptr || key.gen == 0 ||
	    key.slot >= DEFAULT_SHM_LOCAL_TABLE_CAPACITY)
		return;
	try {
		if (auto f = sm->find<detail::shm_local_key_counter>(
			    "bpftime_shm_local_keys");
		    f.first != nullptr) {
			f.first->live_slots[key.slot >> 6].fetch_and(
				~(1ULL << (key.slot & 63u)),
				std::memory_order_release);
		}
	} catch (...) {
	}
}

// Trivially destructible, O(1) direct-indexed .bss table for per-process OS
// file descriptors and mmaps. Fast-path hits take a single array index load
// and execute zero syscalls, zero locks, and zero heap allocations; fork()
// epoch bumps are registered once at library load time.
template <typename Payload,
	  std::size_t Capacity = DEFAULT_SHM_LOCAL_TABLE_CAPACITY>
class shm_local_table {
	static_assert((Capacity & (Capacity - 1)) == 0,
		      "Capacity must be a power of two");
	static_assert(std::is_trivially_destructible_v<Payload>,
		      "Payload must be trivially destructible for exit(0) safety");

	static constexpr uint64_t STATE_IDLE = 0;
	static constexpr uint64_t STATE_BUSY = 1;
	static constexpr uint64_t STATE_READY = 2;
	static constexpr uint64_t STATE_FAILED = 3;

	struct slot {
		std::atomic<uint64_t> tag{ 0 };
		std::atomic<int> fail_errno{ 0 };
		Payload payload{};
	};
	static_assert(std::is_trivially_destructible_v<slot>);

	slot slots_[Capacity]{};

	static uint64_t base_tag_for(uint32_t gen) noexcept
	{
		const uint32_t ep = detail::g_shm_local_fork_epoch.load(
			std::memory_order_acquire);
		return (static_cast<uint64_t>(ep) << 32) |
		       (static_cast<uint64_t>(gen & 0x3fffffffU) << 2);
	}

    public:
	constexpr shm_local_table() noexcept = default;

	Payload *find(shm_local_key key) noexcept
	{
		if (key.gen == 0) [[unlikely]]
			return nullptr;
		const uint64_t base = base_tag_for(key.gen);
		slot &s = slots_[key.slot & (Capacity - 1)];
		if (s.tag.load(std::memory_order_acquire) ==
		    (base | STATE_READY)) [[likely]]
			return &s.payload;
		return nullptr;
	}

	template <typename InitFn, typename CleanupFn = shm_noop_cleanup>
	Payload *get_or_init(shm_local_key key, InitFn &&init_fn,
			     CleanupFn &&on_stale_replace = {}) noexcept
	{
		if (key.gen == 0) [[unlikely]] {
			errno = EINVAL;
			return nullptr;
		}
		const uint64_t base = base_tag_for(key.gen);
		slot &s = slots_[key.slot & (Capacity - 1)];
		uint64_t t = s.tag.load(std::memory_order_acquire);
		if (t == (base | STATE_READY)) [[likely]]
			return &s.payload;
		if (t == (base | STATE_FAILED)) {
			const int err =
				s.fail_errno.load(std::memory_order_relaxed);
			if (err != 0)
				errno = err;
			return nullptr;
		}
		const uint32_t cur_ep = static_cast<uint32_t>(base >> 32);
		const uint32_t slot_ep = static_cast<uint32_t>(t >> 32);
		const uint64_t slot_st = t & 3ULL;
		if (slot_ep == cur_ep && slot_st == STATE_BUSY) {
			errno = EBUSY;
			return nullptr;
		}
		const uint64_t prev_t = t;
		if (!s.tag.compare_exchange_strong(
			    t, base | STATE_BUSY, std::memory_order_acq_rel,
			    std::memory_order_acquire)) {
			if (t == (base | STATE_READY))
				return &s.payload;
			if (t == (base | STATE_FAILED)) {
				const int err = s.fail_errno.load(
					std::memory_order_relaxed);
				if (err != 0)
					errno = err;
			} else {
				errno = EBUSY;
			}
			return nullptr;
		}
		const int caller_errno = errno;
		if ((prev_t & 3ULL) == STATE_READY &&
		    static_cast<uint32_t>(prev_t >> 32) == cur_ep) {
			try {
				on_stale_replace(s.payload);
			} catch (...) {
			}
		}
		s.payload = Payload{};
		Payload tmp{};
		bool ok = false;
		int saved_err = 0;
		try {
			errno = 0;
			ok = init_fn(tmp);
			saved_err = errno;
		} catch (...) {
			ok = false;
			saved_err = errno != 0 ? errno : ENOMEM;
		}
		if (ok) {
			s.payload = tmp;
			s.fail_errno.store(0, std::memory_order_relaxed);
			s.tag.store(base | STATE_READY,
				    std::memory_order_release);
			errno = caller_errno;
			return &s.payload;
		}
		s.fail_errno.store(saved_err, std::memory_order_relaxed);
		s.tag.store(base | STATE_FAILED, std::memory_order_release);
		errno = saved_err != 0 ? saved_err : caller_errno;
		return nullptr;
	}

	template <typename CleanupFn>
	void evict(shm_local_key key, CleanupFn &&cleanup_fn) noexcept
	{
		if (key.gen == 0)
			return;
		const uint64_t base = base_tag_for(key.gen);
		slot &s = slots_[key.slot & (Capacity - 1)];
		uint64_t t = s.tag.load(std::memory_order_acquire);
		if (t != (base | STATE_READY) && t != (base | STATE_FAILED))
			return;
		const bool was_ready = (t == (base | STATE_READY));
		if (s.tag.compare_exchange_strong(
			    t, base | STATE_BUSY, std::memory_order_acq_rel,
			    std::memory_order_acquire)) {
			const int caller_errno = errno;
			if (was_ready) {
				try {
					cleanup_fn(s.payload);
				} catch (...) {
				}
			}
			s.payload = Payload{};
			s.fail_errno.store(0, std::memory_order_relaxed);
			s.tag.store(STATE_IDLE, std::memory_order_release);
			errno = caller_errno;
		}
	}
};

} // namespace bpftime

#endif // _BPFTIME_SHM_SAFE_HPP
