/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, bpftime contributors
 * All rights reserved.
 */
#include <bpf_map/userspace/lpm_trie_map.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <spdlog/spdlog.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <algorithm>
// #include <vector>

namespace bpftime
{

// Constants
static constexpr uint32_t LPM_DATA_SIZE_MAX = 256;
static constexpr uint32_t LPM_DATA_SIZE_MIN = 1;
static constexpr uint32_t LPM_KEY_SIZE_MIN =
	sizeof(uint32_t) + LPM_DATA_SIZE_MIN;
static constexpr uint32_t LPM_KEY_SIZE_MAX =
	sizeof(uint32_t) + LPM_DATA_SIZE_MAX;

lpm_trie_map_impl::lpm_trie_map_impl(
	boost::interprocess::managed_shared_memory &memory,
	unsigned int key_size, unsigned int value_size,
	unsigned int max_entries)
	: _key_size(key_size), _value_size(value_size),
	  _max_entries(max_entries),
	  root(nullptr, node_deleter_type(memory.get_segment_manager())),
	  n_entries(0), memory(memory)
{
	// Validate parameters
	if (key_size < LPM_KEY_SIZE_MIN || key_size > LPM_KEY_SIZE_MAX) {
		SPDLOG_ERROR("LPM Trie key_size ({}) must be between {} and {}",
			     key_size, LPM_KEY_SIZE_MIN, LPM_KEY_SIZE_MAX);
		throw std::invalid_argument("Invalid key_size for LPM Trie");
	}

	if (value_size == 0 || max_entries == 0) {
		SPDLOG_ERROR(
			"LPM Trie value_size ({}) and max_entries ({}) must be > 0",
			value_size, max_entries);
		throw std::invalid_argument(
			"Invalid value_size or max_entries for LPM Trie");
	}

	_data_size = key_size - sizeof(uint32_t); // Subtract prefixlen field
	_max_prefixlen = _data_size * 8; // Maximum prefix length in bits

	SPDLOG_DEBUG(
		"LPM Trie constructed: key_size={}, value_size={}, max_entries={}, data_size={}, max_prefixlen={}",
		_key_size, _value_size, _max_entries, _data_size,
		_max_prefixlen);
}

int lpm_trie_map_impl::extract_bit(const uint8_t *data, size_t index) const
{
	if (!data)
		return 0;
	const size_t max_bits = static_cast<size_t>(_data_size) * 8;
	if (index >= max_bits)
		return 0;
	const uint8_t byte = data[index / 8];
	const uint8_t bit_pos = static_cast<uint8_t>(7 - (index % 8));
	return (byte >> bit_pos) & 0x01;
}

size_t
lpm_trie_map_impl::longest_prefix_match(const lpm_trie_node *node,
					const bpf_lpm_trie_key *key) const
{
	const uint32_t limit = std::min(node->prefixlen, key->prefixlen);
	const uint8_t *node_data = get_node_prefix_data(node);
	size_t matched = 0;
	for (; matched < limit; ++matched) {
		if (extract_bit(node_data, matched) !=
		    extract_bit(key->data, matched))
			break;
	}
	return matched;
}

node_unique_ptr lpm_trie_map_impl::create_node(const void *value,
					       bool is_intermediate)
{
	// Create node in shared memory
	lpm_trie_node *raw_node = memory.construct<lpm_trie_node>(
		boost::interprocess::anonymous_instance)(memory);
	if (!raw_node) {
		return node_unique_ptr(
			nullptr,
			node_deleter_type(memory.get_segment_manager()));
	}

	// Create unique_ptr with proper deleter
	node_unique_ptr node(raw_node,
			     node_deleter_type(memory.get_segment_manager()));

	node->is_intermediate = is_intermediate;

	// Allocate space for prefix data and optionally value data
	size_t total_size = _data_size;
	if (!is_intermediate && value) {
		total_size += _value_size;
	}

	node->data.resize(total_size);

	if (!is_intermediate && value) {
		copy_value_data(boost::interprocess::ipcdetail::to_raw_pointer(
					node.get()),
				value);
	}

	return node;
}

void lpm_trie_map_impl::copy_key_data(lpm_trie_node *node,
				      const bpf_lpm_trie_key *key)
{
	uint8_t *node_data = get_node_prefix_data(node);
	std::memcpy(node_data, key->data, _data_size);
}

void lpm_trie_map_impl::copy_value_data(lpm_trie_node *node, const void *value)
{
	if (node->is_intermediate)
		return;

	uint8_t *value_data = get_node_value_data(node);
	std::memcpy(value_data, value, _value_size);
}

uint8_t *lpm_trie_map_impl::get_node_prefix_data(lpm_trie_node *node)
{
	return node->data.data();
}

uint8_t *lpm_trie_map_impl::get_node_value_data(lpm_trie_node *node)
{
	if (node->is_intermediate)
		return nullptr;
	return node->data.data() + _data_size;
}

const uint8_t *
lpm_trie_map_impl::get_node_prefix_data(const lpm_trie_node *node) const
{
	return node->data.data();
}

const uint8_t *
lpm_trie_map_impl::get_node_value_data(const lpm_trie_node *node) const
{
	if (node->is_intermediate)
		return nullptr;
	return node->data.data() + _data_size;
}

void *lpm_trie_map_impl::elem_lookup(const void *key)
{
	if (!key) {
		SPDLOG_ERROR(
			"LPM Trie elem_lookup failed: key pointer is nullptr");
		errno = EINVAL;
		return nullptr;
	}

	const bpf_lpm_trie_key *lpm_key =
		static_cast<const bpf_lpm_trie_key *>(key);

	if (lpm_key->prefixlen > _max_prefixlen) {
		SPDLOG_ERROR(
			"LPM Trie elem_lookup failed: prefixlen ({}) exceeds maximum ({})",
			lpm_key->prefixlen, _max_prefixlen);
		errno = EINVAL;
		return nullptr;
	}

	lpm_trie_node *node =
		boost::interprocess::ipcdetail::to_raw_pointer(root.get());
	lpm_trie_node *found = nullptr;

	// Walk the trie from root
	while (node) {
		size_t matchlen = longest_prefix_match(node, lpm_key);

		// If we have an exact match for the maximum possible prefix
		if (matchlen == _max_prefixlen) {
			found = node;
			break;
		}

		// If the match is shorter than the node's prefix, we can't go
		// further
		if (matchlen < node->prefixlen) {
			break;
		}

		// Consider this node as a candidate if it's not intermediate
		if (!node->is_intermediate) {
			found = node;
		}

		// Continue traversal
		if (matchlen < lpm_key->prefixlen) {
			int next_bit =
				extract_bit(lpm_key->data, node->prefixlen);
			node = boost::interprocess::ipcdetail::to_raw_pointer(
				node->child[next_bit].get());
		} else {
			break;
		}
	}

	if (!found) {
		errno = ENOENT;
		return nullptr;
	}

	return const_cast<uint8_t *>(get_node_value_data(found));
}

long lpm_trie_map_impl::elem_update(const void *key, const void *value,
				    uint64_t flags)
{
	if (!key || !value) {
		SPDLOG_ERROR(
			"LPM Trie elem_update failed: key ({:p}) or value ({:p}) pointer is nullptr",
			key, value);
		errno = EINVAL;
		return -1;
	}

	// LPM Trie ignores flags (behaves like BPF_ANY)
	if (flags > 2) { // BPF_EXIST = 2
		SPDLOG_ERROR(
			"LPM Trie elem_update failed: invalid flags ({}), only 0-2 supported",
			flags);
		errno = EINVAL;
		return -1;
	}

	const bpf_lpm_trie_key *lpm_key =
		static_cast<const bpf_lpm_trie_key *>(key);

	if (lpm_key->prefixlen > _max_prefixlen) {
		SPDLOG_ERROR(
			"LPM Trie elem_update failed: prefixlen ({}) exceeds maximum ({})",
			lpm_key->prefixlen, _max_prefixlen);
		errno = EINVAL;
		return -1;
	}

	// Check if we're at capacity
	if (n_entries >= _max_entries) {
		errno = ENOSPC;
		return -1;
	}

	// Create new node
	auto new_node = create_node(value, false);
	if (!new_node) {
		errno = ENOMEM;
		return -1;
	}

	new_node->prefixlen = lpm_key->prefixlen;
	copy_key_data(
		boost::interprocess::ipcdetail::to_raw_pointer(new_node.get()),
		lpm_key);

	// If tree is empty, make this the root
	if (!root) {
		root = std::move(new_node);
		n_entries++;
		SPDLOG_TRACE("LPM Trie: Added root node with prefix length {}",
			     lpm_key->prefixlen);
		return 0;
	}

	// Find insertion point
	node_unique_ptr *slot = &root;
	lpm_trie_node *node = nullptr;
	size_t matchlen = 0;

	while (slot->get()) {
		node = boost::interprocess::ipcdetail::to_raw_pointer(
			slot->get());
		matchlen = longest_prefix_match(node, lpm_key);

		if (node->prefixlen != matchlen ||
		    node->prefixlen == lpm_key->prefixlen ||
		    node->prefixlen == _max_prefixlen) {
			break;
		}

		int next_bit = extract_bit(lpm_key->data, node->prefixlen);
		slot = &node->child[next_bit];
	}

	// If slot is empty, insert new node
	if (!slot->get()) {
		*slot = std::move(new_node);
		n_entries++;
		SPDLOG_TRACE("LPM Trie: Added new node with prefix length {}",
			     lpm_key->prefixlen);
		return 0;
	}

	// If exact match, replace the node
	if (node->prefixlen == lpm_key->prefixlen) {
		new_node->child[0] = std::move(node->child[0]);
		new_node->child[1] = std::move(node->child[1]);

		if (node->is_intermediate) {
			// Converting intermediate to real node
		} else {
			// Replacing existing node, don't increment count
			n_entries--;
		}

		*slot = std::move(new_node);
		n_entries++;
		SPDLOG_TRACE("LPM Trie: Replaced node with prefix length {}",
			     lpm_key->prefixlen);
		return 0;
	}

	// Need to create intermediate node or insert as ancestor
	if (matchlen == lpm_key->prefixlen) {
		// New node becomes ancestor
		int next_bit =
			extract_bit(get_node_prefix_data(node), matchlen);
		new_node->child[next_bit] = std::move(*slot);
		*slot = std::move(new_node);
		n_entries++;
		SPDLOG_TRACE(
			"LPM Trie: Added ancestor node with prefix length {}",
			lpm_key->prefixlen);
		return 0;
	}

	// Create intermediate node
	auto im_node = create_node(nullptr, true);
	if (!im_node) {
		errno = ENOMEM;
		return -1;
	}

	im_node->prefixlen = matchlen;
	copy_key_data(
		boost::interprocess::ipcdetail::to_raw_pointer(im_node.get()),
		lpm_key);

	// Determine child placement
	if (extract_bit(lpm_key->data, matchlen)) {
		im_node->child[0] = std::move(*slot);
		im_node->child[1] = std::move(new_node);
	} else {
		im_node->child[0] = std::move(new_node);
		im_node->child[1] = std::move(*slot);
	}

	*slot = std::move(im_node);
	n_entries++;
	SPDLOG_TRACE(
		"LPM Trie: Added intermediate node and new node with prefix length {}",
		lpm_key->prefixlen);
	return 0;
}

long lpm_trie_map_impl::elem_delete(const void *key)
{
	if (!key) {
		SPDLOG_ERROR(
			"LPM Trie elem_delete failed: key pointer is nullptr");
		errno = EINVAL;
		return -1;
	}

	const bpf_lpm_trie_key *lpm_key =
		static_cast<const bpf_lpm_trie_key *>(key);

	if (lpm_key->prefixlen > _max_prefixlen) {
		SPDLOG_ERROR(
			"LPM Trie elem_delete failed: prefixlen ({}) exceeds maximum ({})",
			lpm_key->prefixlen, _max_prefixlen);
		errno = EINVAL;
		return -1;
	}

	// Find the node to delete
	node_unique_ptr *trim = &root;
	node_unique_ptr *trim2 = trim;
	lpm_trie_node *parent = nullptr;
	lpm_trie_node *node = nullptr;

	while (trim->get()) {
		node = boost::interprocess::ipcdetail::to_raw_pointer(
			trim->get());
		size_t matchlen = longest_prefix_match(node, lpm_key);

		if (node->prefixlen != matchlen ||
		    node->prefixlen == lpm_key->prefixlen) {
			break;
		}

		parent = node;
		trim2 = trim;
		int next_bit = extract_bit(lpm_key->data, node->prefixlen);
		trim = &node->child[next_bit];
	}

	if (!node || node->prefixlen != lpm_key->prefixlen ||
	    node->prefixlen != longest_prefix_match(node, lpm_key) ||
	    node->is_intermediate) {
		errno = ENOENT;
		return -1;
	}

	n_entries--;

	// If node has two children, mark as intermediate
	if (node->child[0] && node->child[1]) {
		node->is_intermediate = true;
		// Resize data to remove value part
		node->data.resize(_data_size);
		SPDLOG_TRACE("LPM Trie: Marked node as intermediate");
		return 0;
	}

	// Handle parent cleanup if it's intermediate and child has no siblings
	if (parent && parent->is_intermediate && !node->child[0] &&
	    !node->child[1]) {
		if (boost::interprocess::ipcdetail::to_raw_pointer(
			    trim->get()) ==
		    boost::interprocess::ipcdetail::to_raw_pointer(
			    parent->child[0].get())) {
			*trim2 = std::move(parent->child[1]);
		} else {
			*trim2 = std::move(parent->child[0]);
		}
		SPDLOG_TRACE("LPM Trie: Removed node and promoted sibling");
		return 0;
	}

	// Replace node with its single child (or null)
	if (node->child[0]) {
		*trim = std::move(node->child[0]);
	} else if (node->child[1]) {
		*trim = std::move(node->child[1]);
	} else {
		*trim = node_unique_ptr(
			nullptr,
			node_deleter_type(memory.get_segment_manager()));
	}

	SPDLOG_TRACE("LPM Trie: Removed node with prefix length {}",
		     lpm_key->prefixlen);
	return 0;
}

int lpm_trie_map_impl::map_get_next_key(const void *key, void *next_key)
{
	if (!next_key) {
		SPDLOG_ERROR(
			"LPM Trie map_get_next_key failed: next_key pointer is nullptr");
		errno = EINVAL;
		return -1;
	}

	// Empty trie
	if (!root) {
		errno = ENOENT;
		return -1;
	}

	bpf_lpm_trie_key *next_lpm_key =
		static_cast<bpf_lpm_trie_key *>(next_key);

	// If key is null or invalid, return leftmost node
	if (!key) {
		// Find leftmost non-intermediate node
		lpm_trie_node *node =
			boost::interprocess::ipcdetail::to_raw_pointer(
				root.get());
		while (node) {
			if (!node->is_intermediate) {
				next_lpm_key->prefixlen = node->prefixlen;
				std::memcpy(next_lpm_key->data,
					    get_node_prefix_data(node),
					    _data_size);
				return 0;
			}
			node = node->child[0] ?
				       boost::interprocess::ipcdetail::
					       to_raw_pointer(
						       node->child[0].get()) :
				       boost::interprocess::ipcdetail::
					       to_raw_pointer(
						       node->child[1].get());
		}
		errno = ENOENT;
		return -1;
	}

	// For simplicity, this is a basic implementation
	// A full implementation would do postorder traversal
	errno = ENOENT;
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