/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, bpftime contributors
 * All rights reserved.
 */
#include <bpf_map/userspace/lpm_trie_map.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace bpftime
{

namespace
{
constexpr uint32_t LPM_DATA_SIZE_MAX = 256;
constexpr uint32_t LPM_DATA_SIZE_MIN = 1;
constexpr uint32_t LPM_KEY_SIZE_MIN = sizeof(uint32_t) + LPM_DATA_SIZE_MIN;
constexpr uint32_t LPM_KEY_SIZE_MAX = sizeof(uint32_t) + LPM_DATA_SIZE_MAX;
constexpr uint64_t BPF_ANY_FLAG = 0, BPF_NOEXIST_FLAG = 1, BPF_EXIST_FLAG = 2;
constexpr int LOOKUP_ATTEMPTS = 8;

constexpr size_t align8(size_t n) noexcept
{
	return (n + 7U) & ~size_t(7U);
}
uint32_t load(const uint32_t *p) noexcept
{
	return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
void store(uint32_t *p, uint32_t v) noexcept
{
	__atomic_store_n(p, v, __ATOMIC_RELEASE);
}
} // namespace

lpm_trie_map_impl::lpm_trie_map_impl(const shm_init_ctx &ctx,
				     unsigned int key_size,
				     unsigned int value_size,
				     unsigned int max_entries)
	: _key_size(key_size), _value_size(value_size),
	  _max_entries(max_entries),
	  _pool(bytes_vec_allocator(ctx.segment_manager()))
{
	if (key_size < LPM_KEY_SIZE_MIN || key_size > LPM_KEY_SIZE_MAX) {
		SPDLOG_ERROR("LPM Trie key_size ({}) must be between {} and {}",
			     key_size, LPM_KEY_SIZE_MIN, LPM_KEY_SIZE_MAX);
		throw std::invalid_argument("Invalid key_size for LPM Trie");
	}
	if (value_size == 0 || max_entries == 0 ||
	    max_entries > (NIL - 1) / 2) {
		SPDLOG_ERROR(
			"LPM Trie value_size ({}) must be > 0 and max_entries ({}) in [1, {}]",
			value_size, max_entries, (NIL - 1) / 2);
		throw std::invalid_argument(
			"Invalid value_size or max_entries for LPM Trie");
	}

	_data_size = key_size - sizeof(uint32_t); // Subtract prefixlen field
	_max_prefixlen = _data_size * 8; // Maximum prefix length in bits
	// At most max_entries leaves and max_entries - 1 intermediate nodes.
	_pool_size = 2 * max_entries;
	_val_off = align8(sizeof(node_hdr) + _data_size);
	_stride = align8(_val_off + _value_size);
	_ring_off = size_t(_pool_size) * _stride;
	_pool.resize(_ring_off + size_t(_pool_size) * sizeof(uint32_t));

	SPDLOG_DEBUG(
		"LPM Trie constructed: key_size={}, value_size={}, max_entries={}, data_size={}, max_prefixlen={}, pool_bytes={}",
		_key_size, _value_size, _max_entries, _data_size,
		_max_prefixlen, _pool.size());
}

lpm_trie_map_impl::~lpm_trie_map_impl()
{
}

int lpm_trie_map_impl::extract_bit(const uint8_t *data, size_t index) const
{
	if (index >= static_cast<size_t>(_data_size) * 8)
		return 0;
	const uint8_t byte = __atomic_load_n(&data[index / 8], __ATOMIC_RELAXED);
	return (byte >> (7 - (index % 8))) & 0x01;
}

size_t lpm_trie_map_impl::longest_prefix_match(uint32_t idx, uint32_t prefixlen,
					       const bpf_lpm_trie_key *key) const
{
	const uint32_t limit = std::min(prefixlen, key->prefixlen);
	const uint8_t *node_bits = node_data(idx);
	size_t matched = 0;
	for (; matched < limit; ++matched) {
		if (extract_bit(node_bits, matched) !=
		    extract_bit(key->data, matched))
			break;
	}
	return matched;
}

const bpf_lpm_trie_key *lpm_trie_map_impl::checked_key(const void *key,
						       const char *op) const
{
	if (!key) {
		SPDLOG_ERROR("LPM Trie {} failed: key pointer is nullptr", op);
		errno = EINVAL;
		return nullptr;
	}
	const auto *lpm_key = static_cast<const bpf_lpm_trie_key *>(key);
	if (lpm_key->prefixlen > _max_prefixlen) {
		SPDLOG_ERROR(
			"LPM Trie {} failed: prefixlen ({}) exceeds maximum ({})",
			op, lpm_key->prefixlen, _max_prefixlen);
		errno = EINVAL;
		return nullptr;
	}
	return lpm_key;
}

// Allocation and free run under _write_lock. Each moves one counter with one
// store, so a dead writer at most leaks the node it was handling.
uint32_t lpm_trie_map_impl::alloc_node()
{
	if (_fresh < _pool_size)
		return _fresh++;
	const uint64_t head = _ring_head.load(std::memory_order_relaxed);
	if (head == _ring_tail.load(std::memory_order_relaxed))
		return NIL;
	const uint32_t idx = ring()[head % _pool_size];
	_ring_head.store(head + 1, std::memory_order_relaxed);
	// A lookup may still stand on this node: order the head move before
	// the rewrite, so a lookup that reads new contents also sees the move
	// (see elem_lookup).
	std::atomic_thread_fence(std::memory_order_release);
	return idx;
}

void lpm_trie_map_impl::free_node(uint32_t idx)
{
	// The caller has already unlinked idx: a lookup that sees the new tail
	// also sees the unlink and cannot reach the node.
	const uint64_t tail = _ring_tail.load(std::memory_order_relaxed);
	ring()[tail % _pool_size] = idx;
	_ring_tail.store(tail + 1, std::memory_order_release);
}

uint32_t lpm_trie_map_impl::new_node(const bpf_lpm_trie_key *key,
				     uint32_t prefixlen, const void *value)
{
	const uint32_t idx = alloc_node();
	if (idx == NIL)
		return NIL;
	node_hdr *n = node(idx);
	__atomic_store_n(&n->child[0], NIL, __ATOMIC_RELAXED);
	__atomic_store_n(&n->child[1], NIL, __ATOMIC_RELAXED);
	__atomic_store_n(&n->prefixlen, prefixlen, __ATOMIC_RELAXED);
	__atomic_store_n(&n->flags, value ? 0 : FLAG_IM, __ATOMIC_RELAXED);
	std::memcpy(node_data(idx), key->data, _data_size);
	if (value)
		std::memcpy(node_value(idx), value, _value_size);
	return idx;
}

void *lpm_trie_map_impl::elem_lookup(const void *key)
{
	const bpf_lpm_trie_key *lpm_key = checked_key(key, "elem_lookup");
	if (!lpm_key)
		return nullptr;

	for (int attempt = 0; attempt < LOOKUP_ATTEMPTS; ++attempt) {
		const uint64_t tail = _ring_tail.load(std::memory_order_acquire);
		uint32_t found = NIL;
		uint32_t idx = load(&_root);
		// Bounds and depth checks keep a walk through a recycled node
		// finite; the ring check below then discards its result.
		for (uint32_t depth = 0; idx < _pool_size && depth <= _max_prefixlen;
		     ++depth) {
			node_hdr *n = node(idx);
			const uint32_t plen = load(&n->prefixlen);
			const size_t matchlen =
				longest_prefix_match(idx, plen, lpm_key);
			if (matchlen < plen)
				break;
			if (!(load(&n->flags) & FLAG_IM))
				found = idx;
			if (matchlen == _max_prefixlen ||
			    matchlen >= lpm_key->prefixlen)
				break;
			idx = load(&n->child[extract_bit(lpm_key->data, plen)]);
		}
		// Only nodes freed during the walk sit at ring positions >= tail;
		// none of them was rewritten unless allocation passed tail.
		std::atomic_thread_fence(std::memory_order_acquire);
		if (_ring_head.load(std::memory_order_relaxed) > tail)
			continue;
		if (found == NIL) {
			errno = ENOENT;
			return nullptr;
		}
		// The value lives in the node, so distinct keys never alias and
		// the lookup allocates nothing. The FIFO free ring keeps it
		// readable after the key is deleted until the ring cycles.
		return node_value(found);
	}
	// Freed nodes kept being recycled under the walk: skip like a
	// contended trylock.
	shm_trylock_skips.fetch_add(1, std::memory_order_relaxed);
	return nullptr;
}

long lpm_trie_map_impl::update_locked(const bpf_lpm_trie_key *key,
				      const void *value, uint64_t flags)
{
	uint32_t *slot = &_root;
	uint32_t idx;
	size_t matchlen = 0;
	while ((idx = *slot) != NIL) {
		node_hdr *n = node(idx);
		matchlen = longest_prefix_match(idx, n->prefixlen, key);
		if (n->prefixlen != matchlen || n->prefixlen == key->prefixlen ||
		    n->prefixlen == _max_prefixlen)
			break;
		slot = &n->child[extract_bit(key->data, n->prefixlen)];
	}

	// Same prefix: update the value in place; an intermediate node turns
	// into a real one once its value is written.
	if (idx != NIL && node(idx)->prefixlen == matchlen) {
		node_hdr *n = node(idx);
		const bool im = n->flags & FLAG_IM;
		if (flags == BPF_NOEXIST_FLAG && !im) {
			errno = EEXIST;
			return -1;
		}
		if (flags == BPF_EXIST_FLAG && im) {
			errno = ENOENT;
			return -1;
		}
		if (im && n_entries.load(std::memory_order_relaxed) >=
				  _max_entries) {
			errno = ENOSPC;
			return -1;
		}
		std::memcpy(node_value(idx), value, _value_size);
		if (im) {
			n_entries.fetch_add(1, std::memory_order_relaxed);
			store(&n->flags, 0);
		}
		return 0;
	}

	if (flags == BPF_EXIST_FLAG) {
		errno = ENOENT;
		return -1;
	}
	if (n_entries.load(std::memory_order_relaxed) >= _max_entries) {
		errno = ENOSPC;
		return -1;
	}
	const uint32_t leaf = new_node(key, key->prefixlen, value);
	if (leaf == NIL) {
		errno = ENOMEM;
		return -1;
	}
	uint32_t top = leaf;
	if (idx != NIL && matchlen == key->prefixlen) {
		// The new prefix covers idx: insert it as idx's parent.
		node(leaf)->child[extract_bit(node_data(idx), matchlen)] = idx;
	} else if (idx != NIL) {
		// Split at matchlen with an intermediate node over both.
		top = new_node(key, matchlen, nullptr);
		if (top == NIL) {
			free_node(leaf);
			errno = ENOMEM;
			return -1;
		}
		const int bit = extract_bit(key->data, matchlen);
		node(top)->child[bit] = leaf;
		node(top)->child[!bit] = idx;
	}
	n_entries.fetch_add(1, std::memory_order_relaxed);
	store(slot, top);
	return 0;
}

long lpm_trie_map_impl::elem_update(const void *key, const void *value,
				    uint64_t flags, bool from_syscall)
{
	if (!key || !value) {
		SPDLOG_ERROR(
			"LPM Trie elem_update failed: key ({:p}) or value ({:p}) pointer is nullptr",
			key, value);
		errno = EINVAL;
		return -1;
	}
	if (flags != BPF_ANY_FLAG && flags != BPF_NOEXIST_FLAG &&
	    flags != BPF_EXIST_FLAG) {
		SPDLOG_ERROR("LPM Trie elem_update failed: invalid flags ({})",
			     flags);
		errno = EINVAL;
		return -1;
	}
	const bpf_lpm_trie_key *lpm_key = checked_key(key, "elem_update");
	if (!lpm_key)
		return -1;
	return with_robust_shm_lock(
		true, from_syscall, _write_lock,
		[&]() -> long { return update_locked(lpm_key, value, flags); },
		"lpm_trie");
}

long lpm_trie_map_impl::delete_locked(const bpf_lpm_trie_key *key)
{
	uint32_t *trim = &_root, *trim2 = trim;
	uint32_t parent = NIL, idx;
	size_t matchlen = 0;
	while ((idx = *trim) != NIL) {
		node_hdr *n = node(idx);
		matchlen = longest_prefix_match(idx, n->prefixlen, key);
		if (n->prefixlen != matchlen || n->prefixlen == key->prefixlen)
			break;
		parent = idx;
		trim2 = trim;
		trim = &n->child[extract_bit(key->data, n->prefixlen)];
	}
	if (idx == NIL || node(idx)->prefixlen != key->prefixlen ||
	    matchlen != key->prefixlen || (node(idx)->flags & FLAG_IM)) {
		errno = ENOENT;
		return -1;
	}

	node_hdr *n = node(idx);
	if (n->child[0] != NIL && n->child[1] != NIL) {
		// Two children: keep the node as an intermediate one.
		store(&n->flags, FLAG_IM);
		n_entries.fetch_sub(1, std::memory_order_relaxed);
		return 0;
	}
	if (parent != NIL && (node(parent)->flags & FLAG_IM) &&
	    n->child[0] == NIL && n->child[1] == NIL) {
		// A leaf under an intermediate node: promote its sibling.
		node_hdr *p = node(parent);
		store(trim2, p->child[p->child[0] == idx ? 1 : 0]);
		n_entries.fetch_sub(1, std::memory_order_relaxed);
		free_node(parent);
		free_node(idx);
		return 0;
	}
	store(trim, n->child[0] != NIL ? n->child[0] : n->child[1]);
	n_entries.fetch_sub(1, std::memory_order_relaxed);
	free_node(idx);
	return 0;
}

long lpm_trie_map_impl::elem_delete(const void *key, bool from_syscall)
{
	const bpf_lpm_trie_key *lpm_key = checked_key(key, "elem_delete");
	if (!lpm_key)
		return -1;
	return with_robust_shm_lock(
		true, from_syscall, _write_lock,
		[&]() -> long { return delete_locked(lpm_key); }, "lpm_trie");
}

int lpm_trie_map_impl::map_get_next_key(const void *key, void *next_key)
{
	if (!next_key) {
		SPDLOG_ERROR(
			"LPM Trie map_get_next_key failed: next_key pointer is nullptr");
		errno = EINVAL;
		return -1;
	}
	// Only the first key is supported: the leftmost real node.
	if (key || load(&_root) == NIL) {
		errno = ENOENT;
		return -1;
	}
	auto *next_lpm_key = static_cast<bpf_lpm_trie_key *>(next_key);
	for (int attempt = 0; attempt < LOOKUP_ATTEMPTS; ++attempt) {
		const uint64_t tail = _ring_tail.load(std::memory_order_acquire);
		uint32_t found = NIL, idx = load(&_root);
		for (uint32_t depth = 0; idx < _pool_size && depth <= _max_prefixlen;
		     ++depth) {
			node_hdr *n = node(idx);
			if (!(load(&n->flags) & FLAG_IM)) {
				found = idx;
				break;
			}
			const uint32_t left = load(&n->child[0]);
			idx = left != NIL ? left : load(&n->child[1]);
		}
		if (found != NIL) {
			next_lpm_key->prefixlen = load(&node(found)->prefixlen);
			for (uint32_t i = 0; i < _data_size; ++i)
				next_lpm_key->data[i] = __atomic_load_n(
					&node_data(found)[i], __ATOMIC_RELAXED);
		}
		std::atomic_thread_fence(std::memory_order_acquire);
		if (_ring_head.load(std::memory_order_relaxed) > tail)
			continue;
		if (found == NIL) {
			errno = ENOENT;
			return -1;
		}
		return 0;
	}
	errno = EBUSY;
	return -1;
}

// These operations are not applicable to LPM Trie
long lpm_trie_map_impl::map_push_elem(const void *value, uint64_t flags)
{
	errno = ENOTSUP;
	return -1;
}

long lpm_trie_map_impl::map_pop_elem(void *value)
{
	errno = ENOTSUP;
	return -1;
}

long lpm_trie_map_impl::map_peek_elem(void *value)
{
	errno = ENOTSUP;
	return -1;
}

} // namespace bpftime
