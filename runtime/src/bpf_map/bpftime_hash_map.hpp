#ifndef BPFTIME_HASH_MAP_HPP
#define BPFTIME_HASH_MAP_HPP

#include "map_common_def.hpp"
#include "robust_shm_mutex.hpp"
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>

namespace bpftime
{

using namespace boost::interprocess;

class bpftime_hasher {
    private:
	static inline bool is_prime(size_t n)
	{
		if (n <= 1)
			return false;
		if (n <= 3)
			return true;
		if (n % 2 == 0 || n % 3 == 0)
			return false;
		for (size_t i = 5; i * i <= n; i += 6) {
			if (n % i == 0 || n % (i + 2) == 0)
				return false;
		}
		return true;
	}

    public:
	static inline size_t next_prime(size_t n)
	{
		while (!is_prime(n))
			++n;
		return n;
	}

	static size_t hash_func(const void *key, size_t key_size)
	{
		size_t hash = 0;
		for (size_t i = 0; i < key_size; i++)
			hash = hash * 31 + ((const uint8_t *)key)[i];
		return hash;
	}
};

// Preallocated index-chained hash map shared by HASH and PERCPU_HASH (modeled
// on kernel/bpf/hashtab.c): O(1) slot free-list reuse with zero tombstones,
// lock-free lookup/iteration, and 64-way striped robust_shm_mutex writes.
class bpftime_hash_map {
    public:
	static constexpr size_t NUM_LOCK_STRIPES = 64;

    private:
	static constexpr uint32_t SLOT_FREE = 0, SLOT_FILLED = 1,
				  SLOT_WRITING = 2, END_IDX = UINT32_MAX;
	static constexpr uint64_t FLAG_ANY = 0, FLAG_NOEXIST = 1,
				  FLAG_EXIST = 2, TAG_INC = 1ULL << 32,
				  TAG_MASK = ~0xFFFFFFFFULL;
	struct bucket_hdr { uint32_t head_idx, del_gen; };
	struct slot_hdr { uint32_t state, next_idx; };
	static constexpr size_t align8(size_t n) noexcept { return (n + 7U) & ~size_t(7U); }

	size_t _key_size, _value_size, _slots_offset, _val_offset, _elem_stride;
	size_t _num_buckets, _max_element_count;
	std::atomic<size_t> _count, _alloc_cursor;
	std::atomic<uint64_t> _free_head;
	// One-store publish/unlink: a dead writer leaves chains intact, so recover.
	mutable robust_shm_recovering_mutex _stripes[NUM_LOCK_STRIPES];
	bytes_vec data_buffer;

	uint8_t *raw(size_t off) const noexcept
	{
		return const_cast<uint8_t *>(data_buffer.data()) + off;
	}
	bucket_hdr *bucket(size_t b) const noexcept
	{
		return reinterpret_cast<bucket_hdr *>(raw(b * sizeof(bucket_hdr)));
	}
	slot_hdr *slot(size_t i) const noexcept
	{
		return reinterpret_cast<slot_hdr *>(
			raw(_slots_offset + i * _elem_stride));
	}
	uint32_t load_state(size_t i) const noexcept
	{
		return __atomic_load_n(&slot(i)->state, __ATOMIC_ACQUIRE);
	}
	void store_state(size_t i, uint32_t s) noexcept
	{
		__atomic_store_n(&slot(i)->state, s, __ATOMIC_RELEASE);
	}

	static bool bytes_equal_atomic(const void *a, const void *b,
				       size_t len) noexcept
	{
		const auto *pa = static_cast<const uint8_t *>(a);
		const auto *pb = static_cast<const uint8_t *>(b);
		for (size_t i = 0; i < len; ++i)
			if (__atomic_load_n(&pa[i], __ATOMIC_RELAXED) != pb[i])
				return false;
		return true;
	}

	static void bytes_write_atomic(void *dst, const void *src,
				       size_t len) noexcept
	{
		auto *d = static_cast<uint8_t *>(dst);
		const auto *s = static_cast<const uint8_t *>(src);
		size_t i = 0;
		for (; ((uintptr_t)d & 7U) == 0 && i + 8 <= len; i += 8) {
			uint64_t v = 0;
			if (s != nullptr)
				std::memcpy(&v, s + i, 8);
			__atomic_store_n(reinterpret_cast<uint64_t *>(d + i), v,
					 __ATOMIC_RELAXED);
		}
		// One store for an aligned 4-byte tail (HASH_OF_MAPS inner ids),
		// so a lock-free reader never sees it half-written.
		if (((uintptr_t)(d + i) & 3U) == 0 && i + 4 <= len) {
			uint32_t v = 0;
			if (s != nullptr)
				std::memcpy(&v, s + i, 4);
			__atomic_store_n(reinterpret_cast<uint32_t *>(d + i), v,
					 __ATOMIC_RELAXED);
			i += 4;
		}
		for (; i < len; ++i)
			__atomic_store_n(&d[i], s ? s[i] : uint8_t(0),
					 __ATOMIC_RELAXED);
	}

	static void bytes_read_atomic(void *dst, const void *src,
				      size_t len) noexcept
	{
		auto *d = static_cast<uint8_t *>(dst);
		const auto *s = static_cast<const uint8_t *>(src);
		// One load pairs with the single aligned 4-byte store above.
		if (len == 4 && ((uintptr_t)s & 3U) == 0) {
			const uint32_t v = __atomic_load_n(
				reinterpret_cast<const uint32_t *>(s),
				__ATOMIC_RELAXED);
			std::memcpy(d, &v, 4);
			return;
		}
		for (size_t i = 0; i < len; ++i)
			d[i] = __atomic_load_n(&s[i], __ATOMIC_RELAXED);
	}

	uint32_t alloc_slot(uint32_t &gen) noexcept
	{
		size_t cur = _alloc_cursor.load(std::memory_order_acquire);
		while (cur < _max_element_count) {
			if (_alloc_cursor.compare_exchange_weak(
				    cur, cur + 1, std::memory_order_acq_rel,
				    std::memory_order_acquire)) {
				gen = 1U;
				store_state(cur, (gen << 2) | SLOT_WRITING);
				return static_cast<uint32_t>(cur);
			}
		}
		uint64_t head = _free_head.load(std::memory_order_acquire);
		while (uint32_t(head) != END_IDX) {
			const uint32_t idx = uint32_t(head);
			const uint32_t nxt = __atomic_load_n(
				&slot(idx)->next_idx, __ATOMIC_RELAXED);
			const uint64_t next_head =
				((head & TAG_MASK) + TAG_INC) | nxt;
			if (_free_head.compare_exchange_weak(
				    head, next_head, std::memory_order_acq_rel,
				    std::memory_order_acquire)) {
				gen = (load_state(idx) >> 2) + 1U;
				store_state(idx, (gen << 2) | SLOT_WRITING);
				return idx;
			}
		}
		return END_IDX;
	}

	void free_slot(uint32_t idx) noexcept
	{
		uint64_t head = _free_head.load(std::memory_order_relaxed);
		for (;;) {
			__atomic_store_n(&slot(idx)->next_idx, uint32_t(head),
					 __ATOMIC_RELEASE);
			const uint64_t next_head =
				((head & TAG_MASK) + TAG_INC) | idx;
			if (_free_head.compare_exchange_weak(
				    head, next_head, std::memory_order_release,
				    std::memory_order_relaxed))
				return;
		}
	}

	// value_out, when set, receives the value read before the slot state is
	// rechecked, so a concurrent delete and slot reuse restarts the walk.
	size_t find_slot(const void *key,
			 void *value_out = nullptr) const noexcept
	{
		if (_max_element_count == 0)
			return END_IDX;
		const auto *bh = bucket(
			bpftime_hasher::hash_func(key, _key_size) % _num_buckets);
		for (;;) {
			const uint32_t d1 =
				__atomic_load_n(&bh->del_gen, __ATOMIC_ACQUIRE);
			uint32_t idx = __atomic_load_n(&bh->head_idx,
						       __ATOMIC_ACQUIRE);
			size_t steps = 0;
			bool restart = false;
			while (idx != END_IDX && idx < _max_element_count &&
			       steps++ < _max_element_count) {
				const uint32_t s1 = load_state(idx);
				if ((s1 & 3U) != SLOT_FILLED) {
					restart = true;
					break;
				}
				const bool eq = bytes_equal_atomic(
					get_key(idx), key, _key_size);
				if (eq && value_out != nullptr)
					bytes_read_atomic(value_out,
							  get_value(idx),
							  _value_size);
				const uint32_t nxt = __atomic_load_n(
					&slot(idx)->next_idx, __ATOMIC_ACQUIRE);
				std::atomic_thread_fence(
					std::memory_order_acquire);
				if (s1 != __atomic_load_n(&slot(idx)->state,
							  __ATOMIC_RELAXED)) {
					restart = true;
					break;
				}
				if (eq)
					return idx;
				idx = nxt;
			}
			std::atomic_thread_fence(std::memory_order_acquire);
			if (!restart &&
			    d1 == __atomic_load_n(&bh->del_gen,
						  __ATOMIC_RELAXED))
				return END_IDX;
		}
	}

    public:
	bpftime_hash_map(managed_shared_memory &memory, size_t num_buckets,
			 size_t key_size, size_t value_size)
		: _key_size(key_size), _value_size(value_size),
		  _slots_offset(bpftime_hasher::next_prime(num_buckets) *
				sizeof(bucket_hdr)),
		  _val_offset(sizeof(slot_hdr) + align8(key_size)),
		  _elem_stride(_val_offset + align8(value_size)),
		  _num_buckets(bpftime_hasher::next_prime(num_buckets)),
		  _max_element_count(num_buckets), _count(0), _alloc_cursor(0),
		  _free_head(END_IDX), data_buffer(memory.get_segment_manager())
	{
		data_buffer.resize(
			_slots_offset + _max_element_count * _elem_stride, 0);
		for (size_t b = 0; b < _num_buckets; ++b)
			bucket(b)->head_idx = END_IDX;
	}

	void *get_key(size_t i) const noexcept { return slot(i) + 1; }
	void *get_value(size_t i) const noexcept
	{
		return raw(_slots_offset + i * _elem_stride + _val_offset);
	}
	void *elem_lookup(const void *key) const noexcept
	{
		if (key == nullptr)
			return nullptr;
		const size_t idx = find_slot(key);
		return idx < _max_element_count ? get_value(idx) : nullptr;
	}
	bool elem_lookup_copy(const void *key, void *value_out) const noexcept
	{
		return key != nullptr &&
		       find_slot(key, value_out) < _max_element_count;
	}

	long upsert(const void *key, const void *value, uint64_t flags,
		    bool blocking, size_t slice_offset, size_t slice_size)
	{
		if (key == nullptr || value == nullptr) {
			errno = EINVAL;
			return -1;
		}
		const size_t b =
			bpftime_hasher::hash_func(key, _key_size) % _num_buckets;
		return with_robust_shm_lock(
			true, blocking, _stripes[b % NUM_LOCK_STRIPES],
			[&]() -> long {
				auto *bh = bucket(b);
				const uint32_t head = __atomic_load_n(
					&bh->head_idx, __ATOMIC_RELAXED);
				for (uint32_t idx = head;
				     idx != END_IDX && idx < _max_element_count;
				     idx = __atomic_load_n(&slot(idx)->next_idx,
							   __ATOMIC_RELAXED)) {
					if (!bytes_equal_atomic(get_key(idx),
								key, _key_size))
						continue;
					if (flags == FLAG_NOEXIST) {
						errno = EEXIST;
						return -1L;
					}
					bytes_write_atomic(
						static_cast<uint8_t *>(
							get_value(idx)) +
							slice_offset,
						value, slice_size);
					return 0L;
				}
				if (flags == FLAG_EXIST) {
					errno = ENOENT;
					return -1L;
				}
				size_t cur =
					_count.load(std::memory_order_acquire);
				for (;;) {
					if (cur >= _max_element_count) {
						errno = E2BIG;
						return -1L;
					}
					if (_count.compare_exchange_weak(
						    cur, cur + 1,
						    std::memory_order_acq_rel,
						    std::memory_order_acquire))
						break;
				}
				uint32_t gen = 0;
				const uint32_t claim = alloc_slot(gen);
				if (claim == END_IDX) {
					_count.fetch_sub(
						1, std::memory_order_acq_rel);
					errno = E2BIG;
					return -1L;
				}
				auto *vp =
					static_cast<uint8_t *>(get_value(claim));
				if (slice_size < _value_size)
					bytes_write_atomic(vp, nullptr,
							   _value_size);
				bytes_write_atomic(get_key(claim), key,
						   _key_size);
				bytes_write_atomic(vp + slice_offset, value,
						   slice_size);
				__atomic_store_n(&slot(claim)->next_idx, head,
						 __ATOMIC_RELAXED);
				store_state(claim,
					    ((gen + 1U) << 2) | SLOT_FILLED);
				__atomic_store_n(&bh->head_idx, claim,
						 __ATOMIC_RELEASE);
				return 0L;
			},
			"bpftime_hash_map");
	}

	long erase(const void *key, void *value_out, bool blocking)
	{
		if (key == nullptr) {
			errno = ENOENT;
			return -1;
		}
		const size_t b =
			bpftime_hasher::hash_func(key, _key_size) % _num_buckets;
		return with_robust_shm_lock(
			true, blocking, _stripes[b % NUM_LOCK_STRIPES],
			[&]() -> long {
				auto *bh = bucket(b);
				uint32_t prev = END_IDX;
				for (uint32_t idx = __atomic_load_n(
					     &bh->head_idx, __ATOMIC_RELAXED);
				     idx != END_IDX && idx < _max_element_count;
				     prev = idx,
					      idx = __atomic_load_n(
						      &slot(idx)->next_idx,
						      __ATOMIC_RELAXED)) {
					if (!bytes_equal_atomic(get_key(idx),
								key, _key_size))
						continue;
					const uint32_t nxt = __atomic_load_n(
						&slot(idx)->next_idx,
						__ATOMIC_RELAXED);
					if (value_out != nullptr)
						bytes_read_atomic(
							value_out,
							get_value(idx),
							_value_size);
					__atomic_store_n(
						prev == END_IDX ?
							&bh->head_idx :
							&slot(prev)->next_idx,
						nxt, __ATOMIC_RELEASE);
					__atomic_store_n(
						&bh->del_gen,
						__atomic_load_n(
							&bh->del_gen,
							__ATOMIC_RELAXED) +
							1U,
						__ATOMIC_RELEASE);
					store_state(
						idx,
						(((load_state(idx) >> 2) + 1U)
						 << 2) | SLOT_FREE);
					free_slot(idx);
					_count.fetch_sub(
						1, std::memory_order_acq_rel);
					return 0L;
				}
				errno = ENOENT;
				return -1L;
			},
			"bpftime_hash_map");
	}

	int map_get_next_key(const void *key, void *next_key) const noexcept
	{
		if (next_key == nullptr)
			return errno = EINVAL, -1;
		uint8_t buf[256];
		void *out = _key_size <= sizeof(buf) ? buf : next_key;
		const size_t found = key ? find_slot(key) : END_IDX;
		for (size_t i = found < _max_element_count ? found + 1 : 0;
		     i < _max_element_count; ++i) {
			for (uint32_t s1 = load_state(i);
			     (s1 & 3U) == SLOT_FILLED; s1 = load_state(i)) {
				bytes_read_atomic(out, get_key(i), _key_size);
				std::atomic_thread_fence(
					std::memory_order_acquire);
				if (s1 != __atomic_load_n(&slot(i)->state,
							  __ATOMIC_RELAXED))
					continue;
				if (key && out != key &&
				    std::memcmp(out, key, _key_size) == 0)
					break;
				if (out != next_key)
					std::memcpy(next_key, out, _key_size);
				return 0;
			}
		}
		return errno = ENOENT, -1;
	}

	bool elem_update(const void *k, const void *v)
	{
		return upsert(k, v, FLAG_ANY, true, 0, _value_size) == 0;
	}
	bool elem_delete(const void *k) { return erase(k, nullptr, true) == 0; }
	size_t get_elem_count() const noexcept
	{
		return _count.load(std::memory_order_acquire);
	}
};

} // namespace bpftime

#endif // BPFTIME_HASH_MAP_HPP
