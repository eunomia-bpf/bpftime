/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */

#include "bpf_map/userspace/lru_var_hash_map.hpp"
#include "bpf_map/userspace/per_cpu_array_map.hpp"
#include "bpf_map/userspace/per_cpu_hash_map.hpp"
#include "bpf_map/userspace/stack_trace_map.hpp"
#include "bpftime_shm_internal.hpp"
#include <algorithm>
#include <cstdlib>
#include <string>
#include <type_traits>
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
#include "cuda.h"
#include "bpf_map/gpu/nv_gpu_per_thread_array_map.hpp"
#include "bpf_map/gpu/nv_gpu_shared_array_map.hpp"
#include "bpf_map/gpu/nv_gpu_ringbuf_map.hpp"
#include "bpf_map/gpu_kernel_shared/array_map_kernel_gpu.hpp"
#include "bpf_map/gpu/nv_gpu_shared_hash_map.hpp"
#include "bpf_map/gpu/nv_gpu_array_host_map.hpp"
#include "bpf_map/gpu/nv_gpu_shared_array_host_map.hpp"
#endif
#include "bpf_map/userspace/lpm_trie_map.hpp"
#include <bpf_map/userspace/perf_event_array_map.hpp>
#include <bpf_map/userspace/queue.hpp>
#include <bpf_map/userspace/stack.hpp>
#include <bpf_map/userspace/bloom_filter.hpp>
#include "spdlog/spdlog.h"
#include <cassert>
#include <handler/map_handler.hpp>
#include <bpf_map/userspace/array_map.hpp>
#include <bpf_map/userspace/fix_hash_map.hpp>
#include <bpf_map/userspace/var_hash_map.hpp>
#include <bpf_map/userspace/ringbuf_map.hpp>
#ifdef BPFTIME_BUILD_WITH_LIBBPF
#include <bpf_map/shared/array_map_kernel_user.hpp>
#include <bpf_map/shared/hash_map_kernel_user.hpp>
#include <bpf_map/shared/percpu_array_map_kernel_user.hpp>
#include <bpf_map/shared/perf_event_array_kernel_user.hpp>
#endif
#include <bpf_map/userspace/prog_array.hpp>
#include <bpf_map/userspace/map_in_maps.hpp>
#include <unistd.h>

namespace bpftime
{

using hash_map_impl = fix_size_hash_map_impl;

namespace {
#if defined(__GNUC__) || defined(__clang__)
#define BPFTIME_ALWAYS_INLINE __attribute__((always_inline))
#else
#define BPFTIME_ALWAYS_INLINE
#endif

// Stable per-thread slot for the syscall map-in-map inner fd. The 4-byte value
// is copied here while map_lock is held, so the caller's subsequent memcpy
// cannot read a hash node freed by a concurrent delete.
thread_local int tl_map_in_map_fd_snapshot;

// Per-cpu impls expose separate *_userspace entry points for fd-based
// syscall access; map-in-map impls carry the inner-fd snapshot semantics.
template <typename T>
constexpr bool is_percpu_map_v =
	std::is_same_v<T, per_cpu_array_map_impl> ||
	std::is_same_v<T, per_cpu_hash_map_impl>;

template <typename T>
constexpr bool is_lru_hash_v =
	std::is_same_v<T, lru_var_hash_map_impl>;

template <typename T>
constexpr bool is_map_in_map_v =
	std::is_same_v<T, array_map_of_maps_impl> ||
	std::is_same_v<T, hash_map_of_maps_impl>;

// Compile-time dispatch over the builtin map implementations. Each known
// type resolves the shared-memory pointer inside its case and invokes fn
// with it. The preprocessor guards match map_init exactly, so types absent
// from the current configuration take on_default; unknown types and types
// registered through global_map_ops_table use the same fallback.
template <typename GetPtr, typename Fn, typename Fallback>
BPFTIME_ALWAYS_INLINE inline decltype(auto)
visit_map_impl(bpf_map_type type, GetPtr &&get_ptr, Fn &&fn,
	       Fallback &&on_default)
{
	switch (type) {
	case bpf_map_type::BPF_MAP_TYPE_HASH:
		return fn(static_cast<hash_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_ARRAY: [[likely]]
		return fn(static_cast<array_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_RINGBUF:
		return fn(static_cast<ringbuf_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY:
		return fn(static_cast<perf_event_array_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY:
		return fn(static_cast<per_cpu_array_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH:
		return fn(static_cast<per_cpu_hash_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_STACK_TRACE:
		return fn(static_cast<stack_trace_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_QUEUE:
		return fn(static_cast<queue_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_STACK:
		return fn(static_cast<stack_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_BLOOM_FILTER:
		return fn(static_cast<bloom_filter_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_LRU_HASH:
		return fn(static_cast<lru_var_hash_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_LPM_TRIE:
		return fn(static_cast<lpm_trie_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_ARRAY_OF_MAPS:
		return fn(static_cast<array_map_of_maps_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_HASH_OF_MAPS:
		return fn(static_cast<hash_map_of_maps_impl *>(get_ptr()));
#ifdef BPFTIME_BUILD_WITH_LIBBPF
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY:
		return fn(static_cast<array_map_kernel_user_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_HASH:
		return fn(static_cast<hash_map_kernel_user_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERCPU_ARRAY:
		return fn(static_cast<percpu_array_map_kernel_user_impl *>(
			get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERF_EVENT_ARRAY:
		return fn(static_cast<perf_event_array_kernel_user_impl *>(
			get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_PROG_ARRAY:
		return fn(static_cast<prog_array_map_impl *>(get_ptr()));
#endif
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
	case bpf_map_type::BPF_MAP_TYPE_GPU_HASH_MAP:
		return fn(static_cast<nv_gpu_shared_hash_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP:
		return fn(static_cast<nv_gpu_per_thread_array_map_impl *>(
			get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_MAP:
		return fn(static_cast<nv_gpu_shared_array_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_GPU_RINGBUF_MAP:
		return fn(static_cast<nv_gpu_ringbuf_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP:
		return fn(static_cast<array_map_kernel_gpu_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP:
		return fn(static_cast<nv_gpu_array_host_map_impl *>(get_ptr()));
	case bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP:
		return fn(static_cast<nv_gpu_shared_array_host_map_impl *>(
			get_ptr()));
#endif
	default:
		return on_default();
	}
}
} // namespace

bpftime_map_ops global_map_ops_table[(int)bpf_map_type::BPF_MAP_TYPE_MAX] = {
	{ 0 }
};

uint32_t bpf_map_handler::get_value_size() const
{
	return value_size;
}

uint32_t bpf_map_handler::get_userspace_value_size() const
{
	auto result = value_size;
	if ((type == bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY) ||
	    (type == bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH)) {
		result *= sysconf(_SC_NPROCESSORS_ONLN);
	}
	if (type == bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP ||
	    type == bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP) {
		result *= this->attr.gpu_thread_count;
		SPDLOG_DEBUG(
			"Value size of BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP/HOST is {}",
			result);
	}
	return result;
}

std::optional<ringbuf_map_impl *>
bpf_map_handler::try_get_ringbuf_map_impl() const
{
	if (type != bpf_map_type::BPF_MAP_TYPE_RINGBUF)
		return {};
	return static_cast<ringbuf_map_impl *>(map_impl_ptr.get());
}

std::optional<array_map_impl *> bpf_map_handler::try_get_array_map_impl() const
{
	if (type != bpf_map_type::BPF_MAP_TYPE_ARRAY)
		return {};
	return static_cast<array_map_impl *>(map_impl_ptr.get());
}
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
std::optional<nv_gpu_ringbuf_map_impl *>
bpf_map_handler::try_get_nv_gpu_ringbuf_map_impl() const
{
	if (type != bpf_map_type::BPF_MAP_TYPE_GPU_RINGBUF_MAP)
		return {};
	return static_cast<nv_gpu_ringbuf_map_impl *>(map_impl_ptr.get());
}
#endif
const void *bpf_map_handler::map_lookup_elem(const void *key,
					     bool from_syscall) const
{
	shm_write_window_guard shm_window;
	return visit_map_impl(
		type, [&]() BPFTIME_ALWAYS_INLINE { return map_impl_ptr.get(); },
		[&](auto *impl) -> const void * {
			using impl_t = std::remove_cvref_t<decltype(*impl)>;
			if constexpr (is_lru_hash_v<impl_t>) {
				// LRU enforces BPF_F_RDONLY/WRONLY only on
				// syscall access; helpers pass false.
				return with_robust_shm_lock(
					impl->should_lock, from_syscall, map_lock,
					[&]() -> const void * {
						return impl->elem_lookup(
							key, from_syscall);
					});
			} else if constexpr (is_percpu_map_v<impl_t>) {
				return with_robust_shm_lock(
					impl->should_lock, from_syscall, map_lock,
					[&]() -> const void * {
						if (from_syscall)
							return impl->elem_lookup_userspace(
								key);
						return impl->elem_lookup(key);
					});
			} else if constexpr (is_map_in_map_v<impl_t>) {
				if (!from_syscall)
					return with_robust_shm_lock(
						impl->should_lock, from_syscall, map_lock,
						[&]() -> const void * {
							return impl->elem_lookup(
								key);
						});
				// Syscall path copies the 4-byte inner fd to
				// thread-local storage without map_lock: array
				// slots are never freed, and the hash variant
				// returns an id copied under its slot
				// generation check.
				const int *stored = static_cast<const int *>(
					impl->elem_lookup_userspace(key));
				if (stored == nullptr)
					return nullptr;
				tl_map_in_map_fd_snapshot =
					__atomic_load_n(stored, __ATOMIC_RELAXED);
				return &tl_map_in_map_fd_snapshot;
			}
			return with_robust_shm_lock(
				impl->should_lock, from_syscall, map_lock,
				[&]() -> const void * {
					return impl->elem_lookup(key);
				});
		},
		[&]() -> const void * {
			auto func_ptr =
				global_map_ops_table[(int)type].elem_lookup;
			if (func_ptr)
				return func_ptr(id, key, from_syscall);
			SPDLOG_ERROR("[elem_lookup] Unsupported map type: {}",
				     (int)type);
			return nullptr;
		});
}

long bpf_map_handler::map_update_elem(const void *key, const void *value,
				      uint64_t flags, bool from_syscall) const
{
	shm_write_window_guard shm_window;
	return visit_map_impl(
		type, [&]() BPFTIME_ALWAYS_INLINE { return map_impl_ptr.get(); },
		[&](auto *impl) -> long {
			using impl_t = std::remove_cvref_t<decltype(*impl)>;
			if constexpr (is_map_in_map_v<impl_t>) {
				// Map in maps only support update from syscall
				if (!from_syscall)
					return -EINVAL;
			}
			return with_robust_shm_lock(
				impl->should_lock, from_syscall, map_lock,
				[&]() -> long {
					// LRU enforces BPF_F_RDONLY only on
					// syscall access; helpers pass false.
					// HASH and LPM_TRIE lock internally:
					// try_lock on helper paths.
					if constexpr (is_lru_hash_v<impl_t> ||
						      std::is_same_v<
							      impl_t,
							      hash_map_impl> ||
						      std::is_same_v<
							      impl_t,
							      lpm_trie_map_impl>)
						return impl->elem_update(
							key, value, flags,
							from_syscall);
					else if constexpr (
						is_percpu_map_v<impl_t>) {
						if (from_syscall)
							return impl->elem_update_userspace(
								key, value,
								flags);
						return impl->elem_update(
							key, value, flags);
					}
					return impl->elem_update(
						key, value, flags);
				});
		},
		[&]() -> long {
			auto func_ptr =
				global_map_ops_table[(int)type].elem_update;
			if (func_ptr)
				return func_ptr(id, key, value, flags,
						from_syscall);
			SPDLOG_ERROR(
				"[map_update_elem] Unsupported map type: {}",
				(int)type);
			return -1;
		});
}

int bpf_map_handler::bpf_map_get_next_key(const void *key, void *next_key,
					  bool from_syscall) const
{
	shm_write_window_guard shm_window;
#if !__linux__ && defined(BPFTIME_BUILD_WITH_LIBBPF)
	// This operation historically guarded the kernel-user and prog-array
	// cases with __linux__, unlike the other map operations. On non-Linux
	// builds keep that eligibility: these types take the extension-table/
	// default path instead of dispatching to their impl.
	if (type == bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY ||
	    type == bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_HASH ||
	    type == bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERCPU_ARRAY ||
	    type == bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERF_EVENT_ARRAY ||
	    type == bpf_map_type::BPF_MAP_TYPE_PROG_ARRAY) {
		auto func_ptr =
			global_map_ops_table[(int)type].map_get_next_key;
		if (func_ptr)
			return func_ptr(id, key, next_key, from_syscall);
		SPDLOG_ERROR(
			"[bpf_map_get_next_key] Unsupported map type: {}",
			(int)type);
		return -1;
	}
#endif
	return visit_map_impl(
		type, [&]() BPFTIME_ALWAYS_INLINE { return map_impl_ptr.get(); },
		[&](auto *impl) -> int {
			using impl_t = std::remove_cvref_t<decltype(*impl)>;
			return with_robust_shm_lock(
				impl->should_lock, from_syscall, map_lock,
				[&]() -> int {
					// LRU enforces BPF_F_WRONLY only on
					// syscall access; helpers pass false.
					if constexpr (is_lru_hash_v<impl_t>)
						return impl->map_get_next_key(
							key, next_key,
							from_syscall);
					return impl->map_get_next_key(
						key, next_key);
				});
		},
		[&]() -> int {
			auto func_ptr =
				global_map_ops_table[(int)type]
					.map_get_next_key;
			if (func_ptr)
				return func_ptr(id, key, next_key,
						from_syscall);
			SPDLOG_ERROR(
				"[bpf_map_get_next_key] Unsupported map type: {}",
				(int)type);
			return -1;
		});
}

long bpf_map_handler::map_delete_elem(const void *key, bool from_syscall) const
{
	shm_write_window_guard shm_window;
	return visit_map_impl(
		type, [&]() BPFTIME_ALWAYS_INLINE { return map_impl_ptr.get(); },
		[&](auto *impl) -> long {
			using impl_t = std::remove_cvref_t<decltype(*impl)>;
			if constexpr (is_map_in_map_v<impl_t>) {
				// Map in maps only support mutation from
				// syscall
				if (!from_syscall)
					return -EINVAL;
			}
			return with_robust_shm_lock(
				impl->should_lock, from_syscall, map_lock,
				[&]() -> long {
					// LRU enforces BPF_F_RDONLY only on
					// syscall access; helpers pass false.
					if constexpr (is_lru_hash_v<impl_t> ||
						      std::is_same_v<
							      impl_t,
							      hash_map_impl> ||
						      std::is_same_v<
							      impl_t,
							      lpm_trie_map_impl>) {
						return impl->elem_delete(
							key, from_syscall);
					} else if constexpr (
						is_percpu_map_v<impl_t>) {
						if (from_syscall)
							return impl->elem_delete_userspace(
								key);
						return impl->elem_delete(key);
					}
					return impl->elem_delete(key);
				});
		},
		[&]() -> long {
			auto func_ptr =
				global_map_ops_table[(int)type].elem_delete;
			if (func_ptr)
				return func_ptr(id, key, from_syscall);
			SPDLOG_ERROR(
				"[bpf_map_delete_elem] Unsupported map type: {}",
				(int)type);
			return -1;
		});
}

static uint64_t get_thread_count(const bpf_map_attr &attr)
{
	const auto &config = bpftime_get_runtime_config();
	return config.has_map_gpu_thread_count ? config.map_gpu_thread_count :
						 attr.gpu_thread_count;
}

int bpf_map_handler::map_init(managed_shared_memory &memory)
{
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
	static CUcontext context;
	static CUdevice device;
#endif

	if (map_impl_ptr.get() != nullptr) {
		return 0;
	}

	const auto init_refcnt = [&]() {
		if (map_refcnt_ptr.get() == nullptr) {
			map_refcnt_ptr = shm_construct<map_refcount_t>(
				memory, boost::interprocess::anonymous_instance,
				1);
		} else {
			map_refcnt_ptr->store(1, std::memory_order_relaxed);
		}
	};

	switch (type) {
	case bpf_map_type::BPF_MAP_TYPE_HASH: {
		map_impl_ptr = shm_construct<hash_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			max_entries, key_size, value_size);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_ARRAY: {
		map_impl_ptr = shm_construct<array_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			value_size, max_entries);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_RINGBUF: {
		auto max_ent = max_entries;
		int pop_cnt = 0;
		while (max_ent) {
			pop_cnt += (max_ent & 1);
			max_ent >>= 1;
		}
		if (pop_cnt != 1) {
			SPDLOG_ERROR(
				"Failed to create ringbuf map, max_entries must be a power of 2, current: {}",
				max_entries);
			return -EINVAL;
		}
		map_impl_ptr = shm_construct<ringbuf_map_impl>(
			memory, boost::interprocess::anonymous_instance,
			max_entries, memory);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY: {
		map_impl_ptr = shm_construct<perf_event_array_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			key_size, value_size, max_entries);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY: {
		map_impl_ptr = shm_construct<per_cpu_array_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			value_size, max_entries);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH: {
		map_impl_ptr = shm_construct<per_cpu_hash_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			key_size, value_size, max_entries);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_QUEUE: {
		map_impl_ptr = shm_construct<queue_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			value_size, max_entries);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_STACK: {
		map_impl_ptr = shm_construct<stack_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			value_size, max_entries);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_BLOOM_FILTER: {
		// For bloom filters, key_size must be 0
		if (key_size != 0) {
			SPDLOG_ERROR("Bloom filter key_size must be 0, got {}",
				     key_size);
			return -EINVAL;
		}
		// Extract nr_hashes from map_extra (lower 4 bits)
		unsigned int nr_hashes =
			static_cast<unsigned int>(attr.map_extra & 0xF);
		if (nr_hashes == 0) {
			nr_hashes = 5; // Default value
		}

		// Use JHASH by default (Linux kernel compatible)
		// Could be made configurable via map_extra upper bits in the
		// future
		BloomHashAlgorithm hash_algo = BloomHashAlgorithm::JHASH;

		map_impl_ptr = shm_construct<bloom_filter_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			value_size, max_entries, nr_hashes, hash_algo);
		init_refcnt();
		return 0;
	}

#ifdef BPFTIME_BUILD_WITH_LIBBPF
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY: {
		map_impl_ptr = shm_construct<array_map_kernel_user_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			attr.kernel_bpf_map_id, value_size, max_entries);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_HASH: {
		map_impl_ptr = shm_construct<hash_map_kernel_user_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			attr.kernel_bpf_map_id);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERCPU_ARRAY: {
		map_impl_ptr = shm_construct<percpu_array_map_kernel_user_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			attr.kernel_bpf_map_id);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERF_EVENT_ARRAY: {
		map_impl_ptr =
			shm_construct<perf_event_array_kernel_user_impl>(
				memory, boost::interprocess::anonymous_instance,
				memory, 4, 4, sysconf(_SC_NPROCESSORS_ONLN),
				attr.kernel_bpf_map_id);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_PROG_ARRAY: {
		map_impl_ptr = shm_construct<prog_array_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			key_size, value_size, max_entries);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_STACK_TRACE: {
		map_impl_ptr = shm_construct<stack_trace_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			key_size, value_size, max_entries);
		init_refcnt();
		return 0;
	}
#endif
	case bpf_map_type::BPF_MAP_TYPE_ARRAY_OF_MAPS: {
		// inner fd is stored as a uint32; reject mismatched value_size so
		// the syscall memcpy path can never over-read/truncate (M13 follow-up).
		if (value_size != sizeof(uint32_t))
			return -EINVAL;
		map_impl_ptr = shm_construct<array_map_of_maps_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			max_entries);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_HASH_OF_MAPS: {
		if (value_size != sizeof(uint32_t))
			return -EINVAL;
		map_impl_ptr = shm_construct<hash_map_of_maps_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			key_size, max_entries, static_cast<uint32_t>(flags));
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_LRU_HASH: {
		map_impl_ptr = shm_construct<lru_var_hash_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			key_size, value_size, max_entries,
			static_cast<uint32_t>(flags));
		init_refcnt();
		return 0;
	}

	// TODO: Move these CUDA sentences to a more appropriate position
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
	case bpf_map_type::BPF_MAP_TYPE_GPU_HASH_MAP: {
		shm_holder.global_shared_memory.set_enable_mock(false);
		if (!device) {
			cuDeviceGet(&device, 0);
#if CUDA_VERSION >= 13000
			// CUDA 13.0+ uses 4-parameter cuCtxCreate_v4
			cuCtxCreate(&context, nullptr, 0, device);
#else
			// CUDA 12.x and earlier use 3-parameter cuCtxCreate
			cuCtxCreate(&context, 0, device);
#endif
			SPDLOG_INFO(
				"CUDA context for thread {} has been set to {:x}",
				gettid(), (uintptr_t)context);
		}
		SPDLOG_INFO(
			"Map {} (nv_gpu_shared_hash_map_impl) shared hashmap",
			name.c_str());
		map_impl_ptr = shm_construct<nv_gpu_shared_hash_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			max_entries, key_size, value_size);
		init_refcnt();
		shm_holder.global_shared_memory.set_enable_mock(true);
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP: {
		shm_holder.global_shared_memory.set_enable_mock(false);
		if (!device) {
			cuDeviceGet(&device, 0);
#if CUDA_VERSION >= 13000
			// CUDA 13.0+ uses 4-parameter cuCtxCreate_v4
			cuCtxCreate(&context, nullptr, 0, device);
#else
			// CUDA 12.x and earlier use 3-parameter cuCtxCreate
			cuCtxCreate(&context, 0, device);
#endif
			SPDLOG_INFO(
				"CUDA context for thread {} has been set to {:x}",
				gettid(), (uintptr_t)context);
		}
		// Allow configuring thread count by environment variables
		auto thread_count = get_thread_count(attr);
		SPDLOG_INFO(
			"Map {} (nv_gpu_per_thread_array_map_impl) has space for thread count {}",
			name.c_str(), thread_count);
		map_impl_ptr = shm_construct<nv_gpu_per_thread_array_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			value_size, max_entries, thread_count);
		init_refcnt();
		shm_holder.global_shared_memory.set_enable_mock(true);
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_MAP: {
		shm_holder.global_shared_memory.set_enable_mock(false);
		if (!device) {
			cuDeviceGet(&device, 0);
#if CUDA_VERSION >= 13000
			cuCtxCreate(&context, nullptr, 0, device);
#else
			cuCtxCreate(&context, 0, device);
#endif
			SPDLOG_INFO(
				"CUDA context for thread {} has been set to {:x}",
				gettid(), (uintptr_t)context);
		}
		SPDLOG_INFO(
			"Map {} (nv_gpu_shared_array_map_impl) shared array",
			name.c_str());
		map_impl_ptr = shm_construct<nv_gpu_shared_array_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			value_size, max_entries);
		init_refcnt();
		shm_holder.global_shared_memory.set_enable_mock(true);
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP: {
		shm_holder.global_shared_memory.set_enable_mock(false);
		if (!device) {
			cuDeviceGet(&device, 0);
#if CUDA_VERSION >= 13000
			cuCtxCreate(&context, nullptr, 0, device);
#else
			cuCtxCreate(&context, 0, device);
#endif
			SPDLOG_INFO(
				"CUDA context for thread {} has been set to {:x}",
				gettid(), (uintptr_t)context);
		}
		SPDLOG_INFO("Map {} (array_map_kernel_gpu_impl) shared array",
			    name.c_str());
		map_impl_ptr = shm_construct<array_map_kernel_gpu_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			attr.kernel_bpf_map_id, value_size, max_entries);
		init_refcnt();
		shm_holder.global_shared_memory.set_enable_mock(true);
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_GPU_RINGBUF_MAP: {
		shm_holder.global_shared_memory.set_enable_mock(false);
		if (!device) {
			cuDeviceGet(&device, 0);
#if CUDA_VERSION >= 13000
			// CUDA 13.0+ uses 4-parameter cuCtxCreate_v4
			cuCtxCreate(&context, nullptr, 0, device);
#else
			// CUDA 12.x and earlier use 3-parameter cuCtxCreate
			cuCtxCreate(&context, 0, device);
#endif
			SPDLOG_INFO(
				"CUDA context for thread {} has been set to {:x}",
				gettid(), (uintptr_t)context);
		}
		auto thread_count = get_thread_count(attr);
		SPDLOG_INFO(
			"Map {} (nv_gpu_ringbuf_map_impl) has space for thread count {}",
			name.c_str(), thread_count);
		const uint64_t requested_thread_count = thread_count;
		while (thread_count > 0) {
			try {
				map_impl_ptr =
					shm_construct<nv_gpu_ringbuf_map_impl>(
						memory,
						boost::interprocess::
							anonymous_instance,
						memory, value_size, max_entries,
						thread_count);
				break;
			} catch (const boost::interprocess::bad_alloc &) {
				if (thread_count <= 1) {
					shm_holder.global_shared_memory
						.set_enable_mock(true);
					throw;
				}
				thread_count = std::max<uint64_t>(
					1, thread_count / 2);
			}
		}
		if (thread_count != requested_thread_count) {
			SPDLOG_WARN(
				"Clamped GPU ringbuf thread count from {} to {} due to shared memory pressure; override via BPFTIME_MAP_GPU_THREAD_COUNT",
				requested_thread_count, thread_count);
		}
		init_refcnt();
		shm_holder.global_shared_memory.set_enable_mock(true);
		return 0;
	}
	// HOST versions: use boost::interprocess + cudaHostRegister (for Tegra)
	case bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP: {
		SPDLOG_INFO(
			"Map {} (nv_gpu_array_host_map_impl) has space for thread count {}",
			name.c_str(), attr.gpu_thread_count);
		map_impl_ptr = shm_construct<nv_gpu_array_host_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			value_size, max_entries, attr.gpu_thread_count);
		init_refcnt();
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP: {
		SPDLOG_INFO(
			"Map {} (nv_gpu_shared_array_host_map_impl) shared array",
			name.c_str());
		map_impl_ptr = shm_construct<nv_gpu_shared_array_host_map_impl>(
			memory, boost::interprocess::anonymous_instance, memory,
			value_size, max_entries);
		init_refcnt();
		return 0;
	}
#endif
	case bpf_map_type::BPF_MAP_TYPE_LPM_TRIE: {
		map_impl_ptr = shm_construct<lpm_trie_map_impl>(
			memory, boost::interprocess::anonymous_instance,
			shm_init_ctx(memory), key_size, value_size,
			max_entries);
		init_refcnt();
		return 0;
	}
	default:
		if (bpftime_get_runtime_config().allow_non_buildin_map_types) {
			SPDLOG_INFO("non-builtin map type: {}", (int)type);
			map_impl_ptr = nullptr;
			auto func_ptr =
				global_map_ops_table[(int)type].alloc_map;
			if (func_ptr) {
				return func_ptr(id, name.c_str(), attr);
			}
			return 0;
		} else {
			SPDLOG_ERROR("[map_init] Unsupported map type: {}",
				     (int)type);
			return -EOPNOTSUPP;
		}
	}
	return 0;
}

void bpf_map_handler::map_free(managed_shared_memory &memory) const
{
	if (map_impl_ptr.get() == nullptr) {
		return;
	}
	void *impl = map_impl_ptr.get();

	if (map_refcnt_ptr.get() != nullptr) {
		auto prev = map_refcnt_ptr->fetch_sub(1,
						      std::memory_order_acq_rel);
		if (prev > 1) {
			map_impl_ptr = nullptr;
			map_refcnt_ptr = nullptr;
			return;
		}
		memory.destroy_ptr(map_refcnt_ptr.get());
		map_refcnt_ptr = nullptr;
	}

	auto run_default_free = [&]() {
		auto func_ptr = global_map_ops_table[(int)type].map_free;
		if (func_ptr)
			func_ptr(id);
		else
			SPDLOG_ERROR("[map_free] Unsupported map type: {}",
				     (int)type);
	};
#if !defined(BPFTIME_BUILD_WITH_LIBBPF)
	// The STACK_TRACE destroy case was historically compiled only with
	// libbpf support. Without it, keep that eligibility and run the
	// extension-table/default path instead of destroying the impl here.
	if (type == bpf_map_type::BPF_MAP_TYPE_STACK_TRACE) {
		run_default_free();
		map_impl_ptr = nullptr;
		map_refcnt_ptr = nullptr;
		return;
	}
#endif
	visit_map_impl(
		type, [&]() BPFTIME_ALWAYS_INLINE { return impl; },
		[&](auto *map_impl) { memory.destroy_ptr(map_impl); },
		run_default_free);
	map_impl_ptr = nullptr;
	map_refcnt_ptr = nullptr;
}
std::optional<stack_trace_map_impl *>
bpf_map_handler::try_get_stack_trace_map_impl() const
{
	if (type != bpf_map_type::BPF_MAP_TYPE_STACK_TRACE)
		return {};
	return static_cast<stack_trace_map_impl *>(map_impl_ptr.get());
}
std::optional<perf_event_array_kernel_user_impl *>
bpf_map_handler::try_get_shared_perf_event_array_map_impl() const
{
	if (type != bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERF_EVENT_ARRAY)
		return {};
	return static_cast<perf_event_array_kernel_user_impl *>(
		map_impl_ptr.get());
}
uint64_t bpf_map_handler::get_gpu_map_max_thread_count() const
{
#if !defined(BPFTIME_ENABLE_CUDA_ATTACH)
	return 0;
#endif

#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
	if (this->type == bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP) {
		return static_cast<nv_gpu_per_thread_array_map_impl *>(
			       map_impl_ptr.get())
			->get_max_thread_count();
	}
	if (this->type == bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_MAP ||
	    this->type == bpf_map_type::BPF_MAP_TYPE_GPU_HASH_MAP) {
		return 1;
	}
	if (this->type == bpf_map_type::BPF_MAP_TYPE_GPU_RINGBUF_MAP) {
		return static_cast<nv_gpu_ringbuf_map_impl *>(
			       map_impl_ptr.get())
			->get_max_thread_count();
	}

#ifdef BPFTIME_BUILD_WITH_LIBBPF
	if (this->type == bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY) {
		return 1;
	}
#endif

	if (this->type == bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP) {
		return static_cast<nv_gpu_array_host_map_impl *>(
			       map_impl_ptr.get())
			->get_max_thread_count();
	}
	if (this->type == bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP) {
		return 1;
	}
#endif

	SPDLOG_DEBUG("Not a GPU map!");
	return 0;
}
void *bpf_map_handler::get_gpu_map_extra_buffer() const
{
#if !defined(BPFTIME_ENABLE_CUDA_ATTACH)
	return nullptr;
#endif

#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
	if (this->type == bpf_map_type::BPF_MAP_TYPE_GPU_HASH_MAP) {
		return (void *)static_cast<nv_gpu_shared_hash_map_impl *>(
			       map_impl_ptr.get())
			->get_gpu_mem_buffer();
	}
	if (this->type == bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP) {
		return (void *)static_cast<nv_gpu_per_thread_array_map_impl *>(
			       map_impl_ptr.get())
			->get_gpu_mem_buffer();
	}
	if (this->type == bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_MAP) {
		return (void *)static_cast<nv_gpu_shared_array_map_impl *>(
			       map_impl_ptr.get())
			->get_gpu_mem_buffer();
	}
	if (this->type == bpf_map_type::BPF_MAP_TYPE_GPU_RINGBUF_MAP) {
		return (void *)static_cast<nv_gpu_ringbuf_map_impl *>(
			       map_impl_ptr.get())
			->get_gpu_mem_buffer();
	}
	if (this->type ==
	    bpf_map_type::BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP) {
		return (void *)static_cast<array_map_kernel_gpu_impl *>(
			       map_impl_ptr.get())
			->try_initialize_for_agent_and_get_mapped_address();
	}
	if (this->type == bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP) {
		return (void *)static_cast<nv_gpu_array_host_map_impl *>(
			       map_impl_ptr.get())
			->get_gpu_mem_buffer();
	}
	if (this->type == bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP) {
		return (void *)static_cast<nv_gpu_shared_array_host_map_impl *>(
			       map_impl_ptr.get())
			->get_gpu_mem_buffer();
	}
#endif

	SPDLOG_WARN("Not a GPU map!");
	return nullptr;
}
int bpftime_register_map_ops(int map_type, bpftime_map_ops *ops)
{
	if (map_type < 0 || map_type >= (int)bpf_map_type::BPF_MAP_TYPE_MAX) {
		SPDLOG_ERROR("Invalid map type: {}", map_type);
		return -1;
	}
	global_map_ops_table[map_type] = *ops;
	return 0;
}

// Queue/stack map helper functions implementation
long bpf_map_handler::map_push_elem(const void *value, uint64_t flags,
				    bool from_syscall) const
{
	shm_write_window_guard shm_window;
	switch (type) {
	case bpf_map_type::BPF_MAP_TYPE_QUEUE: {
		auto impl =
			static_cast<queue_map_impl *>(map_impl_ptr.get());
		return with_robust_shm_lock(
			impl->should_lock, from_syscall, map_lock,
			[&] { return impl->map_push_elem(value, flags); });
	}
	case bpf_map_type::BPF_MAP_TYPE_STACK: {
		auto impl =
			static_cast<stack_map_impl *>(map_impl_ptr.get());
		return with_robust_shm_lock(
			impl->should_lock, from_syscall, map_lock,
			[&] { return impl->map_push_elem(value, flags); });
	}
	case bpf_map_type::BPF_MAP_TYPE_BLOOM_FILTER: {
		auto impl = static_cast<bloom_filter_map_impl *>(
			map_impl_ptr.get());
		return with_robust_shm_lock(
			impl->should_lock, from_syscall, map_lock,
			[&] { return impl->map_push_elem(value, flags); });
	}
	default:
		SPDLOG_ERROR("map_push_elem not supported for map type: {}",
			     (int)type);
		return -ENOTSUP;
	}
}

long bpf_map_handler::map_pop_elem(void *value, bool from_syscall) const
{
	shm_write_window_guard shm_window;
	switch (type) {
	case bpf_map_type::BPF_MAP_TYPE_QUEUE: {
		auto impl =
			static_cast<queue_map_impl *>(map_impl_ptr.get());
		return with_robust_shm_lock(
			impl->should_lock, from_syscall, map_lock,
			[&] { return impl->map_pop_elem(value); });
	}
	case bpf_map_type::BPF_MAP_TYPE_STACK: {
		auto impl =
			static_cast<stack_map_impl *>(map_impl_ptr.get());
		return with_robust_shm_lock(
			impl->should_lock, from_syscall, map_lock,
			[&] { return impl->map_pop_elem(value); });
	}
	default:
		SPDLOG_ERROR("map_pop_elem not supported for map type: {}",
			     (int)type);
		return -ENOTSUP;
	}
}

long bpf_map_handler::map_lookup_and_delete_elem(const void *key, void *value,
						 bool from_syscall) const
{
	shm_write_window_guard shm_window;
	// The kernel dispatches by map type: queue/stack take the keyless pop
	// path, the hash variants (HASH, PERCPU_HASH, LRU_HASH,
	// LRU_PERCPU_HASH, STACK_TRACE) take a keyed lookup-then-delete. See
	// map_lookup_and_delete_elem() in kernel/bpf/syscall.c.
	switch (type) {
	case bpf_map_type::BPF_MAP_TYPE_HASH: {
		auto impl =
			static_cast<hash_map_impl *>(map_impl_ptr.get());
		return with_robust_shm_lock(
			impl->should_lock, from_syscall, map_lock, [&] {
				return impl->lookup_and_delete(
					key, value, from_syscall);
			});
	}
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH: {
		auto impl = static_cast<per_cpu_hash_map_impl *>(
			map_impl_ptr.get());
		return with_robust_shm_lock(
			impl->should_lock, from_syscall, map_lock, [&] {
				return impl->lookup_and_delete_userspace(
					key, value, from_syscall);
			});
	}
	case bpf_map_type::BPF_MAP_TYPE_LRU_HASH: {
		auto impl = static_cast<lru_var_hash_map_impl *>(
			map_impl_ptr.get());
		return with_robust_shm_lock(
			impl->should_lock, from_syscall, map_lock, [&] {
				return impl->lookup_and_delete(
					key, value, from_syscall);
			});
	}
	case bpf_map_type::BPF_MAP_TYPE_STACK_TRACE: {
		auto impl = static_cast<stack_trace_map_impl *>(
			map_impl_ptr.get());
		return with_robust_shm_lock(
			impl->should_lock, from_syscall, map_lock, [&] {
				return impl->lookup_and_delete(
					key, value, from_syscall);
			});
	}
	case bpf_map_type::BPF_MAP_TYPE_QUEUE: {
		auto impl =
			static_cast<queue_map_impl *>(map_impl_ptr.get());
		// Queue and stack take the keyless pop, and both declare
		// should_lock = true: the handler owns their lock.
		return with_robust_shm_lock(
			impl->should_lock, from_syscall, map_lock,
			[&] { return impl->map_pop_elem(value); });
	}
	case bpf_map_type::BPF_MAP_TYPE_STACK: {
		auto impl =
			static_cast<stack_map_impl *>(map_impl_ptr.get());
		return with_robust_shm_lock(
			impl->should_lock, from_syscall, map_lock,
			[&] { return impl->map_pop_elem(value); });
	}
	default:
		SPDLOG_ERROR(
			"map_lookup_and_delete_elem not supported for map type: {}",
			(int)type);
		errno = ENOTSUP;
		return -1;
	}
}

long bpf_map_handler::map_peek_elem(void *value, bool from_syscall) const
{
	switch (type) {
	case bpf_map_type::BPF_MAP_TYPE_QUEUE: {
		auto impl =
			static_cast<queue_map_impl *>(map_impl_ptr.get());
		return with_robust_shm_lock(
			impl->should_lock, from_syscall, map_lock,
			[&] { return impl->map_peek_elem(value); });
	}
	case bpf_map_type::BPF_MAP_TYPE_STACK: {
		auto impl =
			static_cast<stack_map_impl *>(map_impl_ptr.get());
		return with_robust_shm_lock(
			impl->should_lock, from_syscall, map_lock,
			[&] { return impl->map_peek_elem(value); });
	}
	case bpf_map_type::BPF_MAP_TYPE_BLOOM_FILTER: {
		auto impl = static_cast<bloom_filter_map_impl *>(
			map_impl_ptr.get());
		return with_robust_shm_lock(
			impl->should_lock, from_syscall, map_lock,
			[&] { return impl->map_peek_elem(value); });
	}
	default:
		SPDLOG_ERROR("map_peek_elem not supported for map type: {}",
			     (int)type);
		return -ENOTSUP;
	}
}

} // namespace bpftime

#undef BPFTIME_ALWAYS_INLINE
