/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, bpftime contributors
 * All rights reserved.
 */
#ifndef _LPM_TRIE_MAP_HPP
#define _LPM_TRIE_MAP_HPP

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <memory>
#include <cstdint>

namespace bpftime
{

// LPM Trie key structure (matches kernel's bpf_lpm_trie_key)
struct bpf_lpm_trie_key {
	uint32_t prefixlen; // Prefix length in bits
	uint8_t data[]; // Data in network byte order (big endian)
};

// Forward declaration
struct lpm_trie_node;

// LPM Trie node structure
struct lpm_trie_node {
	std::shared_ptr<lpm_trie_node> child[2]; // Left and right children
	uint32_t prefixlen; // Prefix length
	uint32_t flags; // Node flags
	bool is_intermediate; // True if intermediate node

	// Data storage: [prefix_data][value_data]
	// prefix_data: data_size bytes
	// value_data: value_size bytes (only for non-intermediate nodes)
	boost::interprocess::vector<
		uint8_t,
		boost::interprocess::allocator<
			uint8_t,
			boost::interprocess::segment_manager<
				char,
				boost::interprocess::rbtree_best_fit<
					boost::interprocess::mutex_family>,
				boost::interprocess::iset_index>>>
		data;

	lpm_trie_node(boost::interprocess::managed_shared_memory &memory)
		: child{ nullptr, nullptr }, prefixlen(0), flags(0),
		  is_intermediate(false),
		  data(boost::interprocess::allocator<
			  uint8_t,
			  boost::interprocess::segment_manager<
				  char,
				  boost::interprocess::rbtree_best_fit<
					  boost::interprocess::mutex_family>,
				  boost::interprocess::iset_index>>(
			  memory.get_segment_manager()))
	{
	}
};

class lpm_trie_map_impl {
    private:
	using byte_allocator = boost::interprocess::allocator<
		uint8_t, boost::interprocess::segment_manager<
				 char,
				 boost::interprocess::rbtree_best_fit<
					 boost::interprocess::mutex_family>,
				 boost::interprocess::iset_index>>;

	unsigned int _key_size;
	unsigned int _value_size;
	unsigned int _max_entries;
	unsigned int _data_size; // Key data size (key_size - sizeof(prefixlen))
	unsigned int _max_prefixlen; // Maximum prefix length in bits

	std::shared_ptr<lpm_trie_node> root;
	size_t n_entries;

	boost::interprocess::managed_shared_memory &memory;

	// Helper functions
	int extract_bit(const uint8_t *data, size_t index) const;
	size_t longest_prefix_match(const lpm_trie_node *node,
				    const bpf_lpm_trie_key *key) const;
	std::shared_ptr<lpm_trie_node>
	create_node(const void *value, bool is_intermediate = false);
	void copy_key_data(lpm_trie_node *node, const bpf_lpm_trie_key *key);
	void copy_value_data(lpm_trie_node *node, const void *value);
	uint8_t *get_node_prefix_data(lpm_trie_node *node);
	uint8_t *get_node_value_data(lpm_trie_node *node);
	const uint8_t *get_node_prefix_data(const lpm_trie_node *node) const;
	const uint8_t *get_node_value_data(const lpm_trie_node *node) const;

    public:
	static constexpr bool should_lock = true;

	lpm_trie_map_impl(boost::interprocess::managed_shared_memory &memory,
			  unsigned int key_size, unsigned int value_size,
			  unsigned int max_entries);

	void *elem_lookup(const void *key);
	long elem_update(const void *key, const void *value, uint64_t flags);
	long elem_delete(const void *key);
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
		return n_entries;
	}
};

} // namespace bpftime

#endif // _LPM_TRIE_MAP_HPP