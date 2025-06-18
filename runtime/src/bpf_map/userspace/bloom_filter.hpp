/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, bpftime contributors
 * All rights reserved.
 *
 * Implementation for BPF_MAP_TYPE_BLOOM_FILTER mapping both standard syscalls
 * and new helper functions:
 * - BPF_MAP_UPDATE_ELEM -> Push (add element to bloom filter)
 * - BPF_MAP_LOOKUP_ELEM -> Peek (check if element exists in bloom filter)
 * - bpf_map_push_elem -> Push (add element to bloom filter)
 * - bpf_map_peek_elem -> Peek (check if element exists in bloom filter)
 *
 * BLOOM FILTER implements probabilistic membership testing:
 * - Push adds elements to the bloom filter (sets bits)
 * - Peek checks if elements might exist (false positives possible, no false
 * negatives)
 * - No delete operation supported (bloom filters don't support deletion)
 * - Key size must be 0 (bloom filters are key-less, only values)
 */
#ifndef BPFTIME_BLOOM_FILTER_MAP_HPP
#define BPFTIME_BLOOM_FILTER_MAP_HPP

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <bpf/bpf.h>
#include <cstdint>

namespace bpftime
{

// Hash algorithm selection for bloom filter
enum class BloomHashAlgorithm {
	DJB2, // Simple djb2 hash (current implementation)
	JHASH // Jenkins hash (Linux kernel style)
};

class bloom_filter_map_impl {
    private:
	unsigned int _value_size;
	unsigned int _max_entries;
	unsigned int _nr_hashes;
	BloomHashAlgorithm _hash_algorithm;
	uint32_t _hash_seed;
	size_t _bit_array_size_bits;
	size_t _bit_array_size_bytes;
	uint32_t _bit_array_mask;

	// Vector type and allocator for storing bit array
	using byte_allocator = boost::interprocess::allocator<
		uint8_t,
		boost::interprocess::managed_shared_memory::segment_manager>;
	using byte_vector =
		boost::interprocess::vector<uint8_t, byte_allocator>;

	byte_vector bit_array;

	// Internal helpers
	uint32_t hash_value(const void *value, uint32_t hash_index) const;
	void set_bit(uint32_t bit_index);
	bool test_bit(uint32_t bit_index) const;
	size_t calculate_optimal_bit_array_size(unsigned int max_entries,
						unsigned int nr_hashes) const;

    public:
	// Map handler should lock externally
	const static bool should_lock = true;

	/**
	 * @brief Construct a new bloom filter map impl object.
	 * @param memory Managed shared memory segment.
	 * @param value_size Size of each element. Must be > 0.
	 * @param max_entries Approximate maximum number of elements (used for
	 * sizing).
	 * @param nr_hashes Number of hash functions to use (default 5 if 0).
	 * @param hash_algorithm Hash algorithm to use (default DJB2).
	 */
	bloom_filter_map_impl(
		boost::interprocess::managed_shared_memory &memory,
		unsigned int value_size, unsigned int max_entries,
		unsigned int nr_hashes,
		BloomHashAlgorithm hash_algorithm = BloomHashAlgorithm::DJB2);

	/**
	 * @brief Destroy the bloom filter map impl object.
	 */
	~bloom_filter_map_impl() = default;

	// Disable copy/assignment
	bloom_filter_map_impl(const bloom_filter_map_impl &) = delete;
	bloom_filter_map_impl &
	operator=(const bloom_filter_map_impl &) = delete;

	// --- Methods mapping to standard BPF syscall commands ---

	/**
	 * @brief Implements BPF_MAP_LOOKUP_ELEM for Bloom Filter (Peek).
	 * Checks if the element might exist in the bloom filter.
	 * @param key Must be NULL for bloom filters.
	 * @return Pointer to a dummy value if element might exist, nullptr if
	 * definitely not present. Note: For bloom filters, the actual value to
	 * check is passed through the syscall layer in a special way. This
	 * function should not be called directly - use map_peek_elem instead.
	 */
	void *elem_lookup(const void *key);

	/**
	 * @brief Implements BPF_MAP_UPDATE_ELEM for Bloom Filter (Push
	 * operation). Adds an element to the bloom filter.
	 * @param key Must be NULL for bloom filters.
	 * @param value Pointer to the value data to add. Must not be NULL.
	 * @param flags Must be BPF_ANY. Other flags result in -EINVAL.
	 * @return 0 on success. Negative error code on failure.
	 */
	long elem_update(const void *key, const void *value, uint64_t flags);

	/**
	 * @brief Delete operation - not supported for bloom filters.
	 * @param key Ignored.
	 * @return Always returns -EOPNOTSUPP.
	 */
	long elem_delete(const void *key);

	/**
	 * @brief Get next key - not applicable for bloom filters.
	 * @param key Ignored.
	 * @param next_key Ignored.
	 * @return Always returns -EOPNOTSUPP.
	 */
	int map_get_next_key(const void *key, void *next_key);

	// --- Methods mapping to bpf_map helper functions ---

	/**
	 * @brief Implements bpf_map_push_elem helper function.
	 * Adds an element to the bloom filter.
	 * @param value Pointer to the value data to add. Must not be NULL.
	 * @param flags Must be BPF_ANY. Other flags result in -EINVAL.
	 * @return 0 on success. Negative error code on failure.
	 */
	long map_push_elem(const void *value, uint64_t flags);

	/**
	 * @brief Pop operation - not supported for bloom filters.
	 * @param value Ignored.
	 * @return Always returns -EOPNOTSUPP.
	 */
	long map_pop_elem(void *value);

	/**
	 * @brief Implements bpf_map_peek_elem helper function.
	 * Checks if an element might exist in the bloom filter.
	 * @param value Pointer to the value to check.
	 * @return 0 if element might exist, -ENOENT if definitely not present.
	 */
	long map_peek_elem(void *value);

	// --- Helper methods ---
	unsigned int get_value_size() const;
	unsigned int get_max_entries() const;
	unsigned int get_nr_hashes() const;
};

} // namespace bpftime

#endif // BPFTIME_BLOOM_FILTER_MAP_HPP