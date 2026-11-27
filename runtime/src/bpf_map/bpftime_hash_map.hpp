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
	static constexpr uint32_t STRIPE_OP_NONE = 0, STRIPE_OP_INSERT = 1,
				  STRIPE_OP_ERASE = 2, POOL_OP_NONE = 0,
				  POOL_OP_ALLOC = 1, POOL_OP_FREE = 2;
	static constexpr uint64_t FLAG_ANY = 0, FLAG_NOEXIST = 1,
				  FLAG_EXIST = 2;
	struct bucket_hdr { uint32_t head_idx, del_gen; };
	struct slot_hdr { uint32_t state, next_idx; };
	struct stripe_state {
		robust_shm_mutex lock;
		uint32_t op = STRIPE_OP_NONE;
		uint32_t bucket = END_IDX;
		uint32_t slot_idx = END_IDX;
		uint32_t orig_gen = 0;
	};
	static constexpr size_t align8(size_t n) noexcept { return (n + 7U) & ~size_t(7U); }

	size_t _key_size, _value_size, _slots_offset, _val_offset, _elem_stride;
	size_t _num_buckets, _max_element_count;
	std::atomic<size_t> _alloc_cursor;
	uint32_t _free_head = END_IDX;
	mutable robust_shm_mutex _pool_lock;
	uint32_t _pool_op = POOL_OP_NONE, _pool_slot = END_IDX,
		 _pool_next = END_IDX, _pool_gen = 0;
	uint8_t _pool_from_cursor = 0;
	mutable stripe_state _stripes[NUM_LOCK_STRIPES];
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
		for (; i < len; ++i)
			__atomic_store_n(&d[i], s ? s[i] : uint8_t(0),
					 __ATOMIC_RELAXED);
	}

	static void bytes_read_atomic(void *dst, const void *src,
				      size_t len) noexcept
	{
		auto *d = static_cast<uint8_t *>(dst);
		const auto *s = static_cast<const uint8_t *>(src);
		for (size_t i = 0; i < len; ++i)
			d[i] = __atomic_load_n(&s[i], __ATOMIC_RELAXED);
	}

	void recover_pool() noexcept
	{
		const uint32_t op = __atomic_load_n(&_pool_op, __ATOMIC_ACQUIRE);
		if (op == POOL_OP_ALLOC && _pool_slot < _max_element_count) {
			if (_pool_from_cursor) {
				if (_alloc_cursor.load(std::memory_order_relaxed) <=
				    _pool_slot)
					_alloc_cursor.store(
						size_t(_pool_slot) + 1U,
						std::memory_order_release);
			} else if (_free_head == _pool_slot) {
				_free_head = _pool_next;
			}
			if ((load_state(_pool_slot) >> 2) < _pool_gen)
				store_state(_pool_slot,
					    (_pool_gen << 2) | SLOT_WRITING);
		} else if (op == POOL_OP_FREE &&
			   _pool_slot < _max_element_count) {
			if (_free_head != _pool_slot) {
				__atomic_store_n(&slot(_pool_slot)->next_idx,
						 _pool_next, __ATOMIC_RELEASE);
				_free_head = _pool_slot;
			}
			store_state(_pool_slot, (_pool_gen << 2) | SLOT_FREE);
		}
		__atomic_store_n(&_pool_op, POOL_OP_NONE, __ATOMIC_RELEASE);
	}

	long alloc_slot(stripe_state &st, size_t b, uint32_t &gen,
			bool blocking) noexcept
	{
		return with_robust_shm_recover_lock(
			blocking, _pool_lock, [&]() noexcept { recover_pool(); },
			[&]() -> long {
				uint32_t cand = END_IDX, nxt = END_IDX;
				uint8_t from_cur = 0;
				const size_t cur = _alloc_cursor.load(
					std::memory_order_relaxed);
				if (_free_head != END_IDX) {
					cand = _free_head;
					nxt = __atomic_load_n(
						&slot(cand)->next_idx,
						__ATOMIC_RELAXED);
					gen = (load_state(cand) >> 2) + 1U;
				} else if (cur < _max_element_count) {
					cand = static_cast<uint32_t>(cur);
					from_cur = 1;
					gen = 1U;
				} else {
					errno = E2BIG;
					return -1L;
				}
				st.bucket = static_cast<uint32_t>(b);
				st.slot_idx = cand;
				st.orig_gen = gen - 1U;
				__atomic_store_n(&st.op, STRIPE_OP_INSERT,
						 __ATOMIC_RELEASE);
				_pool_slot = cand;
				_pool_next = nxt;
				_pool_gen = gen;
				_pool_from_cursor = from_cur;
				__atomic_store_n(&_pool_op, POOL_OP_ALLOC,
						 __ATOMIC_RELEASE);
				if (from_cur)
					_alloc_cursor.store(
						cur + 1U,
						std::memory_order_release);
				else
					_free_head = nxt;
				store_state(cand, (gen << 2) | SLOT_WRITING);
				__atomic_store_n(&_pool_op, POOL_OP_NONE,
						 __ATOMIC_RELEASE);
				return static_cast<long>(cand);
			});
	}

	void free_slot(uint32_t idx, uint32_t target_gen) noexcept
	{
		(void)with_robust_shm_recover_lock(
			true, _pool_lock, [&]() noexcept { recover_pool(); },
			[&]() -> long {
				const uint32_t cur = load_state(idx);
				if ((cur >> 2) > target_gen ||
				    cur == ((target_gen << 2) | SLOT_FREE))
					return 0L;
				_pool_slot = idx;
				_pool_gen = target_gen;
				_pool_next = _free_head;
				__atomic_store_n(&_pool_op, POOL_OP_FREE,
						 __ATOMIC_RELEASE);
				__atomic_store_n(&slot(idx)->next_idx,
						 _free_head, __ATOMIC_RELEASE);
				_free_head = idx;
				store_state(idx,
					    (target_gen << 2) | SLOT_FREE);
				__atomic_store_n(&_pool_op, POOL_OP_NONE,
						 __ATOMIC_RELEASE);
				return 0L;
			});
	}

	bool bucket_contains_slot(size_t b, uint32_t target) const noexcept
	{
		if (b >= _num_buckets || target >= _max_element_count)
			return false;
		size_t steps = 0;
		for (uint32_t idx = __atomic_load_n(&bucket(b)->head_idx,
						    __ATOMIC_ACQUIRE);
		     idx != END_IDX && idx < _max_element_count &&
		     steps++ < _max_element_count;
		     idx = __atomic_load_n(&slot(idx)->next_idx,
					   __ATOMIC_ACQUIRE)) {
			if (idx == target)
				return true;
		}
		return false;
	}

	void recover_stripe(stripe_state &st) noexcept
	{
		const uint32_t op = __atomic_load_n(&st.op, __ATOMIC_ACQUIRE);
		const uint32_t b = st.bucket;
		const uint32_t idx = st.slot_idx;
		const uint32_t g = st.orig_gen;
		if (op == STRIPE_OP_INSERT && idx < _max_element_count) {
			if (!bucket_contains_slot(b, idx)) {
				store_state(idx,
					    ((g + 1U) << 2) | SLOT_WRITING);
				free_slot(idx, g + 1U);
			}
		} else if (op == STRIPE_OP_ERASE && idx < _max_element_count &&
			   b < _num_buckets) {
			if (!bucket_contains_slot(b, idx)) {
				auto *bh = bucket(b);
				__atomic_store_n(&bh->del_gen,
						 __atomic_load_n(&bh->del_gen,
								 __ATOMIC_RELAXED) +
							 1U,
						 __ATOMIC_RELEASE);
				store_state(idx,
					    ((g + 1U) << 2) | SLOT_WRITING);
				free_slot(idx, g + 1U);
			}
		}
		__atomic_store_n(&st.op, STRIPE_OP_NONE, __ATOMIC_RELEASE);
	}

	size_t find_slot(const void *key) const noexcept
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
		  _max_element_count(num_buckets), _alloc_cursor(0),
		  data_buffer(memory.get_segment_manager())
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

	long upsert(const void *key, const void *value, uint64_t flags,
		    bool blocking, size_t slice_offset, size_t slice_size)
	{
		if (key == nullptr || value == nullptr) {
			errno = EINVAL;
			return -1;
		}
		const size_t b =
			bpftime_hasher::hash_func(key, _key_size) % _num_buckets;
		auto &st = _stripes[b % NUM_LOCK_STRIPES];
		return with_robust_shm_recover_lock(
			blocking, st.lock,
			[&]() noexcept { recover_stripe(st); },
			[&]() -> long {
				auto *bh = bucket(b);
				const uint32_t head = __atomic_load_n(
					&bh->head_idx, __ATOMIC_RELAXED);
				for (uint32_t idx = head;
				     idx != END_IDX && idx < _max_element_count;
				     idx = __atomic_load_n(&slot(idx)->next_idx,
							   __ATOMIC_RELAXED)) {
					if (!bytes_equal_atomic(get_key(idx), key,
								_key_size))
						continue;
					if (flags == FLAG_NOEXIST) {
						errno = EEXIST;
						return -1L;
					}
					bytes_write_atomic(
						static_cast<uint8_t *>(get_value(idx)) +
							slice_offset,
						value, slice_size);
					return 0L;
				}
				if (flags == FLAG_EXIST) {
					errno = ENOENT;
					return -1L;
				}
				uint32_t gen = 0;
				const long claim_res =
					alloc_slot(st, b, gen, blocking);
				if (claim_res < 0)
					return -1L;
				const uint32_t claim =
					static_cast<uint32_t>(claim_res);
				auto *vp = static_cast<uint8_t *>(get_value(claim));
				if (slice_size < _value_size)
					bytes_write_atomic(vp, nullptr, _value_size);
				bytes_write_atomic(get_key(claim), key, _key_size);
				bytes_write_atomic(vp + slice_offset, value,
						   slice_size);
				__atomic_store_n(&slot(claim)->next_idx, head,
						 __ATOMIC_RELAXED);
				store_state(claim, (gen << 2) | SLOT_FILLED);
				__atomic_store_n(&bh->head_idx, claim,
						 __ATOMIC_RELEASE);
				__atomic_store_n(&st.op, STRIPE_OP_NONE,
						 __ATOMIC_RELEASE);
				return 0L;
			});
	}

	long erase(const void *key, void *value_out, bool blocking)
	{
		if (key == nullptr) {
			errno = ENOENT;
			return -1;
		}
		const size_t b =
			bpftime_hasher::hash_func(key, _key_size) % _num_buckets;
		auto &st = _stripes[b % NUM_LOCK_STRIPES];
		return with_robust_shm_recover_lock(
			blocking, st.lock,
			[&]() noexcept { recover_stripe(st); },
			[&]() -> long {
				auto *bh = bucket(b);
				uint32_t prev = END_IDX;
				for (uint32_t idx = __atomic_load_n(&bh->head_idx,
								    __ATOMIC_RELAXED);
				     idx != END_IDX && idx < _max_element_count;
				     prev = idx,
				    idx = __atomic_load_n(&slot(idx)->next_idx,
							  __ATOMIC_RELAXED)) {
					if (!bytes_equal_atomic(get_key(idx), key,
								_key_size))
						continue;
					const uint32_t nxt = __atomic_load_n(
						&slot(idx)->next_idx, __ATOMIC_RELAXED);
					const uint32_t g = load_state(idx) >> 2;
					st.bucket = static_cast<uint32_t>(b);
					st.slot_idx = idx;
					st.orig_gen = g;
					__atomic_store_n(&st.op, STRIPE_OP_ERASE,
							 __ATOMIC_RELEASE);
					__atomic_store_n(prev == END_IDX ?
								 &bh->head_idx :
								 &slot(prev)->next_idx,
							 nxt, __ATOMIC_RELEASE);
					__atomic_store_n(
						&bh->del_gen,
						__atomic_load_n(&bh->del_gen,
								__ATOMIC_RELAXED) +
							1U,
						__ATOMIC_RELEASE);
					if (value_out != nullptr)
						bytes_read_atomic(value_out,
								  get_value(idx),
								  _value_size);
					store_state(idx,
						    ((g + 1U) << 2) | SLOT_WRITING);
					free_slot(idx, g + 1U);
					__atomic_store_n(&st.op, STRIPE_OP_NONE,
							 __ATOMIC_RELEASE);
					return 0L;
				}
				errno = ENOENT;
				return -1L;
			});
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
		size_t n = 0;
		const size_t limit = _alloc_cursor.load(std::memory_order_acquire);
		for (size_t i = 0; i < limit && i < _max_element_count; ++i)
			if ((load_state(i) & 3U) == SLOT_FILLED)
				++n;
		return n;
	}
};

} // namespace bpftime

#endif // BPFTIME_HASH_MAP_HPP
