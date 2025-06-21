#include <bpf_map/userspace/bloom_filter.hpp>
#include <spdlog/spdlog.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <fstream>
#include <chrono>

namespace bpftime
{

// Function to generate a secure random seed
static uint32_t generate_random_seed()
{
	uint32_t seed = 0;

	// Try to read from /dev/urandom first (preferred for security)
	std::ifstream urandom("/dev/urandom", std::ios::binary);
	if (urandom.is_open() &&
	    urandom.read(reinterpret_cast<char *>(&seed), sizeof(seed))) {
		urandom.close();
		SPDLOG_DEBUG("Generated hash seed from /dev/urandom: 0x{:08x}",
			     seed);
		return seed;
	}

	// Fallback to timestamp-based seed if /dev/urandom is not available
	auto now = std::chrono::high_resolution_clock::now();
	auto duration = now.time_since_epoch();
	auto nanoseconds =
		std::chrono::duration_cast<std::chrono::nanoseconds>(duration)
			.count();

	// Mix timestamp with some constants to improve randomness
	seed = static_cast<uint32_t>(nanoseconds) ^
	       static_cast<uint32_t>(nanoseconds >> 32) ^
	       0x9e3779b9; // Golden ratio constant

	SPDLOG_DEBUG("Generated hash seed from timestamp: 0x{:08x}", seed);
	return seed;
}

bloom_filter_map_impl::bloom_filter_map_impl(
	boost::interprocess::managed_shared_memory &memory,
	unsigned int value_size, unsigned int max_entries,
	unsigned int nr_hashes, BloomHashAlgorithm hash_algorithm)
	: _value_size(value_size), _max_entries(max_entries),
	  _nr_hashes(nr_hashes), _hash_algorithm(hash_algorithm),
	  bit_array(byte_allocator(memory.get_segment_manager()))
{
	if (value_size == 0 || max_entries == 0) {
		SPDLOG_ERROR(
			"Bloom filter value_size ({}) or max_entries ({}) cannot be zero",
			value_size, max_entries);
		throw std::invalid_argument(
			"Bloom filter value_size or max_entries cannot be zero");
	}

	if (_nr_hashes == 0 || _nr_hashes > 15) {
		SPDLOG_ERROR(
			"Bloom filter nr_hashes ({}) must be between 1 and 15",
			_nr_hashes);
		throw std::invalid_argument("Invalid number of hash functions");
	}

	// Calculate optimal bit array size
	_bit_array_size_bits =
		calculate_optimal_bit_array_size(max_entries, _nr_hashes);
	_bit_array_size_bytes = (_bit_array_size_bits + 7) / 8; // Round up to
								// bytes

	// Ensure bit array size is a power of 2 for efficient masking
	size_t power_of_2_bits = 1;
	while (power_of_2_bits < _bit_array_size_bits) {
		power_of_2_bits <<= 1;
	}
	_bit_array_size_bits = power_of_2_bits;
	_bit_array_size_bytes = (_bit_array_size_bits + 7) / 8;
	_bit_array_mask = static_cast<uint32_t>(_bit_array_size_bits - 1);

	// Initialize bit array
	bit_array.resize(_bit_array_size_bytes, 0);

	// Initialize hash seed with secure random value
	_hash_seed = generate_random_seed();

	const char *algo_name = (_hash_algorithm == BloomHashAlgorithm::JHASH) ?
					"JHASH" :
					"DJB2";
	SPDLOG_DEBUG(
		"Bloom filter constructed: value_size={}, max_entries={}, nr_hashes={}, bit_array_size_bits={}, bit_array_size_bytes={}, hash_algorithm={}",
		_value_size, _max_entries, _nr_hashes, _bit_array_size_bits,
		_bit_array_size_bytes, algo_name);
}

size_t bloom_filter_map_impl::calculate_optimal_bit_array_size(
	unsigned int max_entries, unsigned int nr_hashes) const
{
	// Optimal bit array size: n * k / ln(2) where n = max_entries, k =
	// nr_hashes We use 7/5 to approximate 1/ln(2) ≈ 1.44
	size_t optimal_size =
		(static_cast<size_t>(max_entries) * nr_hashes * 7) / 5;

	// Ensure minimum size
	if (optimal_size < 64) {
		optimal_size = 64;
	}

	// Ensure it doesn't exceed reasonable limits
	if (optimal_size > (1ULL << 31)) {
		optimal_size = 1ULL << 31;
	}

	return optimal_size;
}

// Jenkins hash implementation (similar to Linux kernel jhash)
static uint32_t jhash(const void *key, uint32_t length, uint32_t initval)
{
	const uint8_t *k = static_cast<const uint8_t *>(key);
	uint32_t a, b, c;

	/* Set up the internal state */
	a = b = c = 0xdeadbeef + length + initval;

	/* Handle most of the key */
	while (length > 12) {
		a += k[0] + (static_cast<uint32_t>(k[1]) << 8) +
		     (static_cast<uint32_t>(k[2]) << 16) +
		     (static_cast<uint32_t>(k[3]) << 24);
		b += k[4] + (static_cast<uint32_t>(k[5]) << 8) +
		     (static_cast<uint32_t>(k[6]) << 16) +
		     (static_cast<uint32_t>(k[7]) << 24);
		c += k[8] + (static_cast<uint32_t>(k[9]) << 8) +
		     (static_cast<uint32_t>(k[10]) << 16) +
		     (static_cast<uint32_t>(k[11]) << 24);

		/* Mix */
		a -= c;
		a ^= ((c << 4) | (c >> 28));
		c += b;
		b -= a;
		b ^= ((a << 6) | (a >> 26));
		a += c;
		c -= b;
		c ^= ((b << 8) | (b >> 24));
		b += a;
		a -= c;
		a ^= ((c << 16) | (c >> 16));
		c += b;
		b -= a;
		b ^= ((a << 19) | (a >> 13));
		a += c;
		c -= b;
		c ^= ((b << 4) | (b >> 28));
		b += a;

		length -= 12;
		k += 12;
	}

	/* Handle the last 11 bytes */
	switch (length) {
	case 12:
		c += static_cast<uint32_t>(k[11]) << 24;
		[[fallthrough]];
	case 11:
		c += static_cast<uint32_t>(k[10]) << 16;
		[[fallthrough]];
	case 10:
		c += static_cast<uint32_t>(k[9]) << 8;
		[[fallthrough]];
	case 9:
		c += k[8];
		[[fallthrough]];
	case 8:
		b += static_cast<uint32_t>(k[7]) << 24;
		[[fallthrough]];
	case 7:
		b += static_cast<uint32_t>(k[6]) << 16;
		[[fallthrough]];
	case 6:
		b += static_cast<uint32_t>(k[5]) << 8;
		[[fallthrough]];
	case 5:
		b += k[4];
		[[fallthrough]];
	case 4:
		a += static_cast<uint32_t>(k[3]) << 24;
		[[fallthrough]];
	case 3:
		a += static_cast<uint32_t>(k[2]) << 16;
		[[fallthrough]];
	case 2:
		a += static_cast<uint32_t>(k[1]) << 8;
		[[fallthrough]];
	case 1:
		a += k[0];
		break;
	case 0:
		return c;
	}

	/* Final mix */
	c ^= b;
	c -= ((b << 14) | (b >> 18));
	a ^= c;
	a -= ((c << 11) | (c >> 21));
	b ^= a;
	b -= ((a << 25) | (a >> 7));
	c ^= b;
	c -= ((b << 16) | (b >> 16));
	a ^= c;
	a -= ((c << 4) | (c >> 28));
	b ^= a;
	b -= ((a << 14) | (a >> 18));
	c ^= b;
	c -= ((b << 24) | (b >> 8));

	return c;
}

uint32_t bloom_filter_map_impl::hash_value(const void *value,
					   uint32_t hash_index) const
{
	uint32_t hash;

	if (_hash_algorithm == BloomHashAlgorithm::JHASH) {
		// Use Jenkins hash with different initval for each hash
		// function
		hash = jhash(value, _value_size, _hash_seed + hash_index);
	} else {
		// Use djb2 algorithm with better seed separation
		// Use different multipliers for each hash function to ensure
		// independence
		hash = _hash_seed + (hash_index * 0x9e3779b9); // Golden ratio
							       // constant
		const uint8_t *data = static_cast<const uint8_t *>(value);

		for (unsigned int i = 0; i < _value_size; i++) {
			hash = ((hash << 5) + hash) + data[i]; // hash * 33 + c
		}

		// Apply additional mixing for better distribution
		hash ^= hash >> 16;
		hash *= 0x85ebca6b;
		hash ^= hash >> 13;
		hash *= 0xc2b2ae35;
		hash ^= hash >> 16;
	}

	return hash & _bit_array_mask;
}

void bloom_filter_map_impl::set_bit(uint32_t bit_index)
{
	size_t byte_index = bit_index / 8;
	size_t bit_offset = bit_index % 8;

	if (byte_index < bit_array.size()) {
		bit_array[byte_index] |= (1 << bit_offset);
	}
}

bool bloom_filter_map_impl::test_bit(uint32_t bit_index) const
{
	size_t byte_index = bit_index / 8;
	size_t bit_offset = bit_index % 8;

	if (byte_index < bit_array.size()) {
		return (bit_array[byte_index] & (1 << bit_offset)) != 0;
	}
	return false;
}

void *bloom_filter_map_impl::elem_lookup(const void *key)
{
	SPDLOG_WARN(
		"elem_lookup called on bloom filter - this should use map_peek_elem instead");
	errno = EOPNOTSUPP;
	return nullptr;
}

long bloom_filter_map_impl::elem_update(const void *key, const void *value,
					uint64_t flags)
{
	if (key != nullptr) {
		SPDLOG_WARN(
			"Bloom filter update called with non-nullptr key, ignoring key.");
	}

	if (value == nullptr) {
		SPDLOG_ERROR(
			"Bloom filter update failed: value pointer is nullptr");
		errno = EINVAL;
		return -1;
	}

	if (flags != BPF_ANY) {
		SPDLOG_ERROR(
			"Bloom filter update failed: invalid flags ({}), only BPF_ANY supported",
			flags);
		errno = EINVAL;
		return -1;
	}

	// Add element to bloom filter by setting corresponding bits
	for (unsigned int i = 0; i < _nr_hashes; i++) {
		uint32_t bit_index = hash_value(value, i);
		set_bit(bit_index);
	}

	SPDLOG_TRACE("Bloom filter elem_update: success");
	return 0;
}

long bloom_filter_map_impl::elem_delete(const void *key)
{
	SPDLOG_TRACE("Bloom filter elem_delete: operation not supported");
	errno = EOPNOTSUPP;
	return -1;
}

int bloom_filter_map_impl::map_get_next_key(const void *key, void *next_key)
{
	errno = EOPNOTSUPP;
	return -1;
}

long bloom_filter_map_impl::map_push_elem(const void *value, uint64_t flags)
{
	if (value == nullptr) {
		SPDLOG_ERROR(
			"Bloom filter map_push_elem failed: value pointer is nullptr");
		errno = EINVAL;
		return -1;
	}

	if (flags != BPF_ANY) {
		SPDLOG_ERROR(
			"Bloom filter map_push_elem failed: invalid flags ({}), only BPF_ANY supported",
			flags);
		errno = EINVAL;
		return -1;
	}

	// Add element to bloom filter by setting corresponding bits
	for (unsigned int i = 0; i < _nr_hashes; i++) {
		uint32_t bit_index = hash_value(value, i);
		set_bit(bit_index);
	}

	SPDLOG_TRACE("Bloom filter map_push_elem: success");
	return 0;
}

long bloom_filter_map_impl::map_pop_elem(void *value)
{
	SPDLOG_TRACE("Bloom filter map_pop_elem: operation not supported");
	errno = EOPNOTSUPP;
	return -1;
}

long bloom_filter_map_impl::map_peek_elem(void *value)
{
	if (value == nullptr) {
		SPDLOG_ERROR(
			"Bloom filter map_peek_elem failed: value pointer is nullptr");
		errno = EINVAL;
		return -1;
	}

	// Check if element might exist in bloom filter
	for (unsigned int i = 0; i < _nr_hashes; i++) {
		uint32_t bit_index = hash_value(value, i);
		if (!test_bit(bit_index)) {
			SPDLOG_TRACE(
				"Bloom filter map_peek_elem: element definitely not present");
			errno = ENOENT;
			return -1;
		}
	}

	SPDLOG_TRACE("Bloom filter map_peek_elem: element might be present");
	return 0;
}

unsigned int bloom_filter_map_impl::get_value_size() const
{
	return _value_size;
}

unsigned int bloom_filter_map_impl::get_max_entries() const
{
	return _max_entries;
}

unsigned int bloom_filter_map_impl::get_nr_hashes() const
{
	return _nr_hashes;
}

} // namespace bpftime