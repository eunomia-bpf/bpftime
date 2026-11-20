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
#include <sched.h>
#include "bpftime_shm_internal.hpp"
#include "platform_utils.hpp"
// #include <vector>

namespace bpftime
{

namespace
{
struct SpinLockGuard {
	pthread_spinlock_t &lock;
	explicit SpinLockGuard(pthread_spinlock_t &l) : lock(l)
	{
		pthread_spin_lock(&lock);
	}
	~SpinLockGuard()
	{
		pthread_spin_unlock(&lock);
	}
	SpinLockGuard(const SpinLockGuard &) = delete;
	SpinLockGuard &operator=(const SpinLockGuard &) = delete;
};
} // namespace

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
	  n_entries(0), memory(memory), slot_stride(0),
	  cpu_to_slot(memory.get_segment_manager()),
	  lookup_scratch(memory.get_segment_manager())
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

	// Pad each slot to max_align_t so every slot start is naturally
	// aligned for the typed BPF load that consumes the returned pointer.
	constexpr size_t kAlign = alignof(max_align_t);
	slot_stride =
		(static_cast<size_t>(_value_size) + kAlign - 1) & ~(kAlign - 1);

	// Build a deterministic BPF-CPU -> compact-slot map from this
	// process's cpuset. Every process in a container shares the same
	// cpuset, so the mapping is identical across processes that map the
	// same SHM segment; the per-CPU execution lock then guarantees one
	// writer per slot. Sizing slots by the allowed CPU count (not
	// BPFTIME_MAX_CPU_COUNT) keeps a large value_size affordable in the
	// default segment. This runs at map creation, never on the BPF hot
	// path, so it may use libc freely.
	cpu_to_slot.assign(BPFTIME_MAX_CPU_COUNT, -1);
	cpu_set_t allowed;
	CPU_ZERO(&allowed);
	size_t online = 0;
	if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
		for (int cpu = 0; cpu < BPFTIME_MAX_CPU_COUNT; ++cpu) {
			if (CPU_ISSET(cpu, &allowed))
				cpu_to_slot[cpu] = static_cast<int>(online++);
		}
	}
	// Fallback if the affinity query is unavailable: assume dense CPU ids
	// up to the online processor count.
	if (online == 0) {
		long n = sysconf(_SC_NPROCESSORS_ONLN);
		online = static_cast<size_t>(n > 0 ? n : 1);
		for (size_t cpu = 0;
		     cpu < online &&
		     cpu < static_cast<size_t>(BPFTIME_MAX_CPU_COUNT);
		     ++cpu)
			cpu_to_slot[cpu] = static_cast<int>(cpu);
	}
	lookup_scratch.resize(online * slot_stride);

	// Initialize process-shared spin lock
	int rc = pthread_spin_init(&mapSpin, PTHREAD_PROCESS_SHARED);
	if (rc != 0) {
		SPDLOG_ERROR("LPM Trie: pthread_spin_init failed: {}", rc);
		throw std::runtime_error("Failed to initialize spin lock");
	}

	SPDLOG_DEBUG(
		"LPM Trie constructed: key_size={}, value_size={}, max_entries={}, data_size={}, max_prefixlen={}",
		_key_size, _value_size, _max_entries, _data_size,
		_max_prefixlen);
}

lpm_trie_map_impl::~lpm_trie_map_impl()
{
	pthread_spin_destroy(&mapSpin);
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
	SpinLockGuard lock(mapSpin);
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

	// Copy the matched node value out before releasing mapSpin, then hand
	// back a stable buffer. The destination depends on the caller:
	//
	//  - BPF helper context (a pinned BPF CPU exists): copy into the
	//    SHM-resident per-CPU scratch. The returned pointer must lie inside
	//    the global SHM segment or the uBPF runtime bounds check rejects the
	//    BPF dereference. The guard guarantees the CPU index is in range and
	//    BPF execution is serialized per CPU, so the slot has no concurrent
	//    writer. This path stays allocation-free so it is safe when BPF fires
	//    while libc holds an arena lock.
	//  - bpf() syscall context (no pinned BPF CPU): this path is not
	//    CPU-serialized, so a per-CPU slot could be aliased by another
	//    thread. Keep the original per-thread buffer instead, which isolates
	//    concurrent syscall callers and may allocate freely because a direct
	//    bpf() syscall is not a libc allocation callback.
	const uint8_t *src = get_node_value_data(found);
	if (!src) {
		errno = ENOENT;
		return nullptr;
	}
	if (bpftime_get_current_bpf_cpu() >= 0) {
		int cpu = my_sched_getcpu();
		if (cpu < 0 ||
		    cpu >= static_cast<int>(BPFTIME_MAX_CPU_COUNT)) {
			// Defensive: the guard normally keeps the BPF CPU in
			// range, so this is unreachable on the BPF path.
			errno = EINVAL;
			return nullptr;
		}
		int slot = cpu_to_slot[cpu];
		if (slot < 0) {
			// The pinned CPU is outside this process's cpuset; the
			// guard/cpuset should prevent this on the BPF path.
			errno = EINVAL;
			return nullptr;
		}
		uint8_t *dst = lookup_scratch.data() +
			       static_cast<size_t>(slot) * slot_stride;
		// Make the slot (not just the segment) bound explicit: the
		// whole value must stay inside the pre-allocated scratch.
		if (dst + _value_size >
		    lookup_scratch.data() + lookup_scratch.size()) {
			errno = EINVAL;
			return nullptr;
		}
		std::memcpy(dst, src, _value_size);
		return dst;
	}

	alignas(std::max_align_t) static thread_local uint8_t inline_value[256]
#if defined(__linux__)
		__attribute__((tls_model("initial-exec")))
#endif
		= {};
	if (_value_size <= sizeof(inline_value)) {
		std::memcpy(inline_value, src, _value_size);
		return inline_value;
	}
	static thread_local std::vector<uint8_t> tl_value;
	tl_value.resize(_value_size);
	std::memcpy(tl_value.data(), src, _value_size);
	return tl_value.data();
}

long lpm_trie_map_impl::elem_update(const void *key, const void *value,
				    uint64_t flags)
{
	SpinLockGuard lock(mapSpin);
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

	// Helper lambdas
	auto is_exact_match = [&](const lpm_trie_node *n) -> bool {
		return n && n->prefixlen == lpm_key->prefixlen &&
		       longest_prefix_match(n, lpm_key) == lpm_key->prefixlen;
	};

	// If tree is empty
	if (!root) {
		if (flags == 2 /*BPF_EXIST*/) {
			errno = ENOENT;
			return -1;
		}
		if (n_entries >= _max_entries) {
			errno = ENOSPC;
			return -1;
		}
		auto new_node = create_node(value, false);
		if (!new_node) {
			errno = ENOMEM;
			return -1;
		}
		new_node->prefixlen = lpm_key->prefixlen;
		copy_key_data(boost::interprocess::ipcdetail::to_raw_pointer(
				      new_node.get()),
			      lpm_key);
		root = std::move(new_node);
		n_entries++;
		SPDLOG_TRACE("LPM Trie: Added root node with prefix length {}",
			     lpm_key->prefixlen);
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
				if (n_entries >= _max_entries) {
					errno = ENOSPC;
					return -1;
				}
				node->is_intermediate = false;
				node->data.resize(_data_size + _value_size);
				n_entries++;
			}
			copy_value_data(node, value);
			return 0;
		}
		// Not exact: split at current matchlen into intermediate
		if (flags == 2 /*EXIST*/) {
			errno = ENOENT;
			return -1;
		}
		if (n_entries >= _max_entries) {
			errno = ENOSPC;
			return -1;
		}
		auto new_node = create_node(value, false);
		if (!new_node) {
			errno = ENOMEM;
			return -1;
		}
		new_node->prefixlen = lpm_key->prefixlen;
		copy_key_data(boost::interprocess::ipcdetail::to_raw_pointer(
				      new_node.get()),
			      lpm_key);
		auto im_node = create_node(nullptr, true);
		if (!im_node) {
			errno = ENOMEM;
			return -1;
		}
		im_node->prefixlen = matchlen;
		copy_key_data(boost::interprocess::ipcdetail::to_raw_pointer(
				      im_node.get()),
			      lpm_key);
		if (extract_bit(lpm_key->data, matchlen)) {
			im_node->child[0] = std::move(*slot);
			im_node->child[1] = std::move(new_node);
		} else {
			im_node->child[0] = std::move(new_node);
			im_node->child[1] = std::move(*slot);
		}
		*slot = std::move(im_node);
		n_entries++;
		return 0;
	}

	// Case 2: slot is empty -> pure insert
	if (!slot->get()) {
		if (flags == 2 /*EXIST*/) {
			errno = ENOENT;
			return -1;
		}
		if (n_entries >= _max_entries) {
			errno = ENOSPC;
			return -1;
		}
		auto new_node = create_node(value, false);
		if (!new_node) {
			errno = ENOMEM;
			return -1;
		}
		new_node->prefixlen = lpm_key->prefixlen;
		copy_key_data(boost::interprocess::ipcdetail::to_raw_pointer(
				      new_node.get()),
			      lpm_key);
		*slot = std::move(new_node);
		n_entries++;
		SPDLOG_TRACE("LPM Trie: Added new node with prefix length {}",
			     lpm_key->prefixlen);
		return 0;
	}

	// Case 3: need to insert as ancestor of current node
	if (matchlen == lpm_key->prefixlen) {
		if (flags == 2 /*EXIST*/) {
			errno = ENOENT;
			return -1;
		}
		if (n_entries >= _max_entries) {
			errno = ENOSPC;
			return -1;
		}
		auto new_node = create_node(value, false);
		if (!new_node) {
			errno = ENOMEM;
			return -1;
		}
		new_node->prefixlen = lpm_key->prefixlen;
		copy_key_data(boost::interprocess::ipcdetail::to_raw_pointer(
				      new_node.get()),
			      lpm_key);
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

	// Case 4: split by creating intermediate node between
	if (flags == 2 /*EXIST*/) {
		errno = ENOENT;
		return -1;
	}
	if (n_entries >= _max_entries) {
		errno = ENOSPC;
		return -1;
	}
	auto new_node = create_node(value, false);
	if (!new_node) {
		errno = ENOMEM;
		return -1;
	}
	new_node->prefixlen = lpm_key->prefixlen;
	copy_key_data(
		boost::interprocess::ipcdetail::to_raw_pointer(new_node.get()),
		lpm_key);

	auto im_node = create_node(nullptr, true);
	if (!im_node) {
		errno = ENOMEM;
		return -1;
	}
	im_node->prefixlen = matchlen;
	copy_key_data(
		boost::interprocess::ipcdetail::to_raw_pointer(im_node.get()),
		lpm_key);
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
	SpinLockGuard lock(mapSpin);
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

	// Traverse to find the slot pointing to the node to delete
	node_unique_ptr *slot = &root;
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
	if (n_entries > 0) {
		n_entries--;
	}
	SPDLOG_TRACE("LPM Trie: Marked node as intermediate (logical delete)");
	return 0;
}

int lpm_trie_map_impl::map_get_next_key(const void *key, void *next_key)
{
	SpinLockGuard lock(mapSpin);
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