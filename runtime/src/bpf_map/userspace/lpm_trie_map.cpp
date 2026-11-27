/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, bpftime contributors
 * All rights reserved.
 */
#include <bpf_map/userspace/lpm_trie_map.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <spdlog/spdlog.h>
#include <cerrno>
#include <cstddef>
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

lpm_trie_map_impl::lpm_trie_map_impl(const shm_init_ctx &ctx,
				     unsigned int key_size,
				     unsigned int value_size,
				     unsigned int max_entries)
	: _key_size(key_size), _value_size(value_size),
	  _max_entries(max_entries),
	  roots_(make_roots(ctx.segment_manager(),
			    std::make_index_sequence<NUM_SHARDS + 1>{})),
	  segment_manager_(ctx.segment_manager())
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

lpm_trie_map_impl::~lpm_trie_map_impl()
{
}

size_t lpm_trie_map_impl::shard_for(const bpf_lpm_trie_key *key) const noexcept
{
	if (_data_size < 4 || key->prefixlen < 32)
		return NUM_SHARDS;
	uint32_t h = 0;
	std::memcpy(&h, key->data, sizeof(h));
	h ^= h >> 16;
	return ((h * 2654435761u) >> 26) & (NUM_SHARDS - 1);
}

bool lpm_trie_map_impl::reserve_slot(size_t shard) noexcept
{
	size_t cur = n_entries.load(std::memory_order_relaxed);
	while (cur < _max_entries &&
	       !n_entries.compare_exchange_weak(cur, cur + 1,
						std::memory_order_acq_rel,
						std::memory_order_relaxed)) {
	}
	if (cur >= _max_entries)
		return false;
	if (shard == NUM_SHARDS)
		short_entries_.fetch_add(1, std::memory_order_release);
	return true;
}

void lpm_trie_map_impl::release_slot(size_t shard) noexcept
{
	if (shard == NUM_SHARDS)
		short_entries_.fetch_sub(1, std::memory_order_release);
	n_entries.fetch_sub(1, std::memory_order_acq_rel);
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
	segment_manager_type *sm = segment_manager_.get();
	// Create node in shared memory via position-independent segment_manager
	lpm_trie_node *raw_node = sm->construct<lpm_trie_node>(
		boost::interprocess::anonymous_instance)(sm);
	if (!raw_node) {
		return node_unique_ptr(nullptr, node_deleter_type(sm));
	}

	// Create unique_ptr with proper deleter
	node_unique_ptr node(raw_node, node_deleter_type(sm));

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

void *lpm_trie_map_impl::find_value_locked(size_t shard,
					   const bpf_lpm_trie_key *lpm_key)
{
	lpm_trie_node *node = boost::interprocess::ipcdetail::to_raw_pointer(
		roots_[shard].get());
	lpm_trie_node *found = nullptr;
	while (node) {
		size_t matchlen = longest_prefix_match(node, lpm_key);
		if (matchlen == _max_prefixlen) {
			found = node;
			break;
		}
		if (matchlen < node->prefixlen)
			break;
		if (!node->is_intermediate)
			found = node;
		if (matchlen >= lpm_key->prefixlen)
			break;
		node = boost::interprocess::ipcdetail::to_raw_pointer(
			node->child[extract_bit(lpm_key->data, node->prefixlen)]
				.get());
	}
	if (!found)
		return nullptr;
	// Resolve the inline value while the stripe is still held so a
	// concurrent logical delete cannot flip is_intermediate underneath us.
	// The value lives in the node itself rather than a per-thread copy-out
	// buffer: a second lookup must not clobber a first result, and this path
	// is reachable from a malloc callback while libc holds an arena lock, so
	// no allocation is permitted here. Logical delete never frees node
	// storage, so the returned address stays valid after the stripe unlocks.
	return get_node_value_data(found);
}

void *lpm_trie_map_impl::elem_lookup(const void *key, bool from_syscall)
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

	const size_t shard = shard_for(lpm_key);
	bool outer_locked = false;
	bool fallback_skipped = false;
	// The long-prefix stripe stays held for the whole resolution. When the
	// key may also match a shorter (<32-bit) prefix stored in the fallback
	// stripe, that stripe is acquired nested in the fixed order
	// long -> fallback; it cannot deadlock because the fallback stripe never
	// takes a long-prefix stripe. Helper trylock on either stays non-blocking.
	void *result = with_robust_shm_lock(
		true, from_syscall, stripes_[shard],
		[&]() -> void * {
			outer_locked = true;
			void *value = find_value_locked(shard, lpm_key);
			if (value || shard == NUM_SHARDS ||
			    short_entries_.load(std::memory_order_acquire) == 0)
				return value;
			bool fallback_locked = false;
			void *short_value = with_robust_shm_lock(
				true, from_syscall, stripes_[NUM_SHARDS],
				[&]() -> void * {
					fallback_locked = true;
					return find_value_locked(NUM_SHARDS,
								 lpm_key);
				},
				"lpm_trie_map");
			if (!fallback_locked)
				fallback_skipped = true;
			return short_value;
		},
		"lpm_trie_map");

	// Helper trylock miss on either stripe: skip the event while the
	// gateway has already restored host errno.
	if (!outer_locked || fallback_skipped)
		return nullptr;
	if (!result) {
		if (from_syscall)
			errno = ENOENT;
		return nullptr;
	}
	return result;
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

	// Validate flags
	if (flags != 0 /*BPF_ANY*/ && flags != 1 /*BPF_NOEXIST*/ &&
	    flags != 2 /*BPF_EXIST*/) {
		SPDLOG_ERROR("LPM Trie elem_update failed: invalid flags ({})",
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

	const size_t shard = shard_for(lpm_key);
	return with_robust_shm_lock(
		true, from_syscall, stripes_[shard],
		[&]() -> long {
			return update_shard(shard, lpm_key, value, flags);
		},
		"lpm_trie_map");
}

long lpm_trie_map_impl::update_shard(size_t shard,
				     const bpf_lpm_trie_key *lpm_key,
				     const void *value, uint64_t flags)
{
	node_unique_ptr &root = roots_[shard];

	// Rolls a reserved entry slot back when the insertion fails or throws
	// before committing. Stack-only: no heap allocation and no syscall.
	struct slot_rollback {
		lpm_trie_map_impl &self;
		const size_t shard;
		bool armed;
		slot_rollback(lpm_trie_map_impl &s, size_t sh)
			: self(s), shard(sh), armed(true)
		{
		}
		~slot_rollback()
		{
			if (armed)
				self.release_slot(shard);
		}
		void commit()
		{
			armed = false;
		}
	};

	// Helper lambdas
	auto is_exact_match = [&](const lpm_trie_node *n) -> bool {
		return n && n->prefixlen == lpm_key->prefixlen &&
		       longest_prefix_match(n, lpm_key) == lpm_key->prefixlen;
	};

	auto alloc_keyed = [&](uint32_t plen, const void *val,
			       bool im) -> node_unique_ptr {
		auto n = create_node(val, im);
		if (!n)
			return n;
		n->prefixlen = plen;
		copy_key_data(
			boost::interprocess::ipcdetail::to_raw_pointer(n.get()),
			lpm_key);
		return n;
	};
	auto split_slot = [&](node_unique_ptr *slot,
			      size_t matchlen) -> long {
		if (flags == 2 /*EXIST*/)
			return errno = ENOENT, -1;
		if (!reserve_slot(shard))
			return errno = ENOSPC, -1;
		slot_rollback rollback(*this, shard);
		auto new_node = alloc_keyed(lpm_key->prefixlen, value, false);
		auto im_node = alloc_keyed(matchlen, nullptr, true);
		if (!new_node || !im_node) {
			errno = ENOMEM;
			return -1;
		}
		if (extract_bit(lpm_key->data, matchlen)) {
			im_node->child[0] = std::move(*slot);
			im_node->child[1] = std::move(new_node);
		} else {
			im_node->child[0] = std::move(new_node);
			im_node->child[1] = std::move(*slot);
		}
		*slot = std::move(im_node);
		rollback.commit();
		return 0;
	};

	// If tree is empty
	if (!root) {
		if (flags == 2 /*BPF_EXIST*/) {
			errno = ENOENT;
			return -1;
		}
		if (!reserve_slot(shard)) {
			errno = ENOSPC;
			return -1;
		}
		slot_rollback rollback(*this, shard);
		auto new_node = alloc_keyed(lpm_key->prefixlen, value, false);
		if (!new_node) {
			errno = ENOMEM;
			return -1;
		}
		root = std::move(new_node);
		rollback.commit();
		return 0;
	}

	// Walk to find position
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

	// Case 1: same prefix length slot
	if (slot->get() && node->prefixlen == lpm_key->prefixlen) {
		bool exact = is_exact_match(node);
		if (exact) {
			bool exists_real = !node->is_intermediate;
			if (flags == 1 /*NOEXIST*/) {
				errno = EEXIST;
				return -1;
			}
			if (flags == 2 /*EXIST*/ && !exists_real) {
				errno = ENOENT;
				return -1;
			}
			if (!exists_real) {
				if (!reserve_slot(shard)) {
					errno = ENOSPC;
					return -1;
				}
				slot_rollback rollback(*this, shard);
				// Grow storage before marking the node real: a
				// throwing resize leaves it intermediate and the
				// reserved slot is rolled back.
				node->data.resize(_data_size + _value_size);
				node->is_intermediate = false;
				rollback.commit();
			}
			copy_value_data(node, value);
			return 0;
		}
		return split_slot(slot, matchlen);
	}

	// Case 2: slot is empty -> pure insert
	if (!slot->get()) {
		if (flags == 2 /*EXIST*/) {
			errno = ENOENT;
			return -1;
		}
		if (!reserve_slot(shard)) {
			errno = ENOSPC;
			return -1;
		}
		slot_rollback rollback(*this, shard);
		auto new_node = alloc_keyed(lpm_key->prefixlen, value, false);
		if (!new_node) {
			errno = ENOMEM;
			return -1;
		}
		*slot = std::move(new_node);
		rollback.commit();
		return 0;
	}

	// Case 3: need to insert as ancestor of current node
	if (matchlen == lpm_key->prefixlen) {
		if (flags == 2 /*EXIST*/) {
			errno = ENOENT;
			return -1;
		}
		if (!reserve_slot(shard)) {
			errno = ENOSPC;
			return -1;
		}
		slot_rollback rollback(*this, shard);
		auto new_node = alloc_keyed(lpm_key->prefixlen, value, false);
		if (!new_node) {
			errno = ENOMEM;
			return -1;
		}
		int next_bit =
			extract_bit(get_node_prefix_data(node), matchlen);
		new_node->child[next_bit] = std::move(*slot);
		*slot = std::move(new_node);
		rollback.commit();
		return 0;
	}

	// Case 4: split by creating intermediate node between
	return split_slot(slot, matchlen);
}

long lpm_trie_map_impl::elem_delete(const void *key, bool from_syscall)
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

	const size_t shard = shard_for(lpm_key);
	return with_robust_shm_lock(
		true, from_syscall, stripes_[shard],
		[&]() -> long { return delete_shard(shard, lpm_key); },
		"lpm_trie_map");
}

long lpm_trie_map_impl::delete_shard(size_t shard,
				     const bpf_lpm_trie_key *lpm_key)
{
	// Traverse to find the slot pointing to the node to delete
	node_unique_ptr *slot = &roots_[shard];
	lpm_trie_node *node = nullptr;
	while (slot->get()) {
		node = boost::interprocess::ipcdetail::to_raw_pointer(
			slot->get());
		size_t matchlen = longest_prefix_match(node, lpm_key);
		if (node->prefixlen != matchlen ||
		    node->prefixlen == lpm_key->prefixlen) {
			break;
		}
		int next_bit = extract_bit(lpm_key->data, node->prefixlen);
		slot = &node->child[next_bit];
	}

	if (!slot->get() || node->prefixlen != lpm_key->prefixlen ||
	    longest_prefix_match(node, lpm_key) != node->prefixlen ||
	    node->is_intermediate) {
		errno = ENOENT;
		return -1;
	}

	// Mark node as intermediate (logical deletion), avoid structural
	// changes
	node->is_intermediate = true;
	node->data.resize(_data_size);
	release_slot(shard);
	return 0;
}

int lpm_trie_map_impl::map_get_next_key(const void *key, void *next_key,
					bool from_syscall)
{
	if (!next_key) {
		SPDLOG_ERROR(
			"LPM Trie map_get_next_key failed: next_key pointer is nullptr");
		errno = EINVAL;
		return -1;
	}

	if (key || n_entries.load(std::memory_order_acquire) == 0) {
		errno = ENOENT;
		return -1;
	}

	bpf_lpm_trie_key *next_lpm_key =
		static_cast<bpf_lpm_trie_key *>(next_key);
	for (size_t s = 0; s <= NUM_SHARDS; ++s) {
		bool locked = false;
		int rc = with_robust_shm_lock(
			true, from_syscall, stripes_[s],
			[&]() -> int {
				locked = true;
				for (lpm_trie_node *node = boost::interprocess::
					     ipcdetail::to_raw_pointer(
						     roots_[s].get());
				     node;
				     node = boost::interprocess::ipcdetail::
					     to_raw_pointer(
						     (node->child[0] ?
							      node->child[0] :
							      node->child[1])
							     .get())) {
					if (node->is_intermediate)
						continue;
					next_lpm_key->prefixlen =
						node->prefixlen;
					std::memcpy(next_lpm_key->data,
						    get_node_prefix_data(node),
						    _data_size);
					return 0;
				}
				return 1;
			},
			"lpm_trie_map");
		if (!locked || rc == 0)
			return locked ? 0 : rc;
	}
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