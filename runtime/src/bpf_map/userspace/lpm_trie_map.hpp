/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, bpftime contributors
 * All rights reserved.
 */
#ifndef _LPM_TRIE_MAP_HPP
#define _LPM_TRIE_MAP_HPP

#include "bpftime_shm_safe.hpp"
#include "bpf_map/map_common_def.hpp"
#include "robust_shm_mutex.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace bpftime
{

// LPM Trie key structure (matches kernel's bpf_lpm_trie_key)
struct bpf_lpm_trie_key {
	uint32_t prefixlen; // Prefix length in bits
	uint8_t data[]; // Data in network byte order (big endian)
};

// Kernel-style LPM trie (kernel/bpf/lpm_trie.c) over a pool of
// 2 * max_entries fixed nodes preallocated at creation and linked by uint32_t
// indices. Every intermediate node keeps two children, so the pool always
// holds the tree.
//
// Lookups take no lock. A writer publishes a fully built node or unlinks one
// with a single store, so a reader walking concurrently always sees a
// well-formed tree. Freed nodes queue in a FIFO ring and are recycled only
// after every node freed before them, so a value pointer returned by a lookup
// stays valid until the queue cycles. A lookup notes the ring's free position
// before the walk; if allocation passed it by the end, a node freed during the
// walk may have been rewritten, and the lookup retries, returning NULL after a
// few attempts like a trylock skip. No counter is left half-updated by a dead
// writer, so no reader ever waits for a repair.
//
// Writes serialize on one robust_shm_recovering_mutex (try_lock on helper
// paths). A writer that dies at any step leaves the tree intact; it leaks at
// most the two nodes it was linking or freeing and may leave n_entries one
// high, so nothing is poisoned.
class lpm_trie_map_impl {
    private:
	struct node_hdr {
		uint32_t child[2];
		uint32_t prefixlen;
		uint32_t flags;
	};
	static constexpr uint32_t NIL = UINT32_MAX, FLAG_IM = 1;

	unsigned int _key_size;
	unsigned int _value_size;
	unsigned int _max_entries;
	unsigned int _data_size; // Key data size (key_size - sizeof(prefixlen))
	unsigned int _max_prefixlen; // Maximum prefix length in bits
	uint32_t _pool_size; // 2 * max_entries nodes
	size_t _val_off, _stride, _ring_off;

	uint32_t _root = NIL;
	std::atomic<size_t> n_entries{ 0 };
	// Written only under _write_lock: never-used nodes are handed out first,
	// then freed ones in FIFO order through the ring.
	uint32_t _fresh = 0;
	std::atomic<uint64_t> _ring_head{ 0 }, _ring_tail{ 0 };
	mutable robust_shm_recovering_mutex _write_lock;
	bytes_vec _pool;

	node_hdr *node(uint32_t i) const noexcept
	{
		return reinterpret_cast<node_hdr *>(
			const_cast<uint8_t *>(_pool.data()) + size_t(i) * _stride);
	}
	uint8_t *node_data(uint32_t i) const noexcept
	{
		return reinterpret_cast<uint8_t *>(node(i)) + sizeof(node_hdr);
	}
	uint8_t *node_value(uint32_t i) const noexcept
	{
		return reinterpret_cast<uint8_t *>(node(i)) + _val_off;
	}
	uint32_t *ring() const noexcept
	{
		return reinterpret_cast<uint32_t *>(
			const_cast<uint8_t *>(_pool.data()) + _ring_off);
	}

	int extract_bit(const uint8_t *data, size_t index) const;
	size_t longest_prefix_match(uint32_t idx, uint32_t prefixlen,
				    const bpf_lpm_trie_key *key) const;
	const bpf_lpm_trie_key *checked_key(const void *key,
					    const char *op) const;
	uint32_t alloc_node();
	void free_node(uint32_t idx);
	uint32_t new_node(const bpf_lpm_trie_key *key, uint32_t prefixlen,
			  const void *value);
	long update_locked(const bpf_lpm_trie_key *key, const void *value,
			   uint64_t flags);
	long delete_locked(const bpf_lpm_trie_key *key);

    public:
	// Writes take _write_lock themselves; lookups take none.
	static constexpr bool should_lock = false;

	lpm_trie_map_impl(const shm_init_ctx &ctx, unsigned int key_size,
			  unsigned int value_size, unsigned int max_entries);
	~lpm_trie_map_impl();

	void *elem_lookup(const void *key);
	long elem_update(const void *key, const void *value, uint64_t flags,
			 bool from_syscall = true);
	long elem_delete(const void *key, bool from_syscall = true);
	int map_get_next_key(const void *key, void *next_key);

	// LPM Trie specific operations (not used but required by interface)
	long map_push_elem(const void *value, uint64_t flags);
	long map_pop_elem(void *value);
	long map_peek_elem(void *value);

	unsigned int get_key_size() const
	{
		return _key_size;
	}
	unsigned int get_value_size() const
	{
		return _value_size;
	}
	unsigned int get_max_entries() const
	{
		return _max_entries;
	}
	size_t get_current_entries() const
	{
		return n_entries.load(std::memory_order_relaxed);
	}
};

} // namespace bpftime

#endif // _LPM_TRIE_MAP_HPP
