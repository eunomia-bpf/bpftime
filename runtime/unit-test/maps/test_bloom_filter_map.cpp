#include "catch2/catch_test_macros.hpp"
#include "catch2/catch_message.hpp"

#include "bpf_map/userspace/bloom_filter.hpp"
#include "../common_def.hpp"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <vector>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <random>
#include <set>
#include <string>

using namespace bpftime;
using namespace boost::interprocess;

TEST_CASE("Bloom Filter Constructor Validation", "[bloom_filter][constructor]")
{
	const char *SHARED_MEMORY_NAME = "BloomFilterTestShmCatch2";
	const size_t SHARED_MEMORY_SIZE = 1024 * 1024;

	// RAII for shared memory segment removal
	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	// Test invalid value_size (0)
	REQUIRE_THROWS_AS(bloom_filter_map_impl(shm, 0, 10, 5),
			  std::invalid_argument);

	// Test invalid max_entries (0)
	REQUIRE_THROWS_AS(bloom_filter_map_impl(shm, 4, 0, 5),
			  std::invalid_argument);

	// Test invalid nr_hashes (0)
	REQUIRE_THROWS_AS(bloom_filter_map_impl(shm, 4, 10, 0),
			  std::invalid_argument);

	// Test invalid nr_hashes (too large)
	REQUIRE_THROWS_AS(bloom_filter_map_impl(shm, 4, 10, 16),
			  std::invalid_argument);

	// Test valid construction
	REQUIRE_NOTHROW(bloom_filter_map_impl(shm, 4, 10, 5));
}

TEST_CASE("Bloom Filter Basic Operations", "[bloom_filter][basic]")
{
	const char *SHARED_MEMORY_NAME = "BloomFilterBasicTestShm";
	const size_t SHARED_MEMORY_SIZE = 1024 * 1024;

	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	bloom_filter_map_impl bf(shm, sizeof(int), 100, 5);

	SECTION("Basic Properties")
	{
		REQUIRE(bf.get_value_size() == sizeof(int));
		REQUIRE(bf.get_max_entries() == 100);
		REQUIRE(bf.get_nr_hashes() == 5);
	}

	SECTION("Push and Peek Operations")
	{
		int value1 = 42;
		int value2 = 123;
		int value3 = 999;

		// Add elements to bloom filter
		REQUIRE(bf.map_push_elem(&value1, BPF_ANY) == 0);
		REQUIRE(bf.map_push_elem(&value2, BPF_ANY) == 0);

		// Check if elements might exist (should return 0 for "might
		// exist")
		REQUIRE(bf.map_peek_elem(&value1) == 0);
		REQUIRE(bf.map_peek_elem(&value2) == 0);

		// Check for element that was not added (might return 0 due to
		// false positive, or -1) We can't guarantee the result for
		// value3, but it should not crash
		int result = bf.map_peek_elem(&value3);
		REQUIRE((result == 0 || result == -1));
	}

	SECTION("elem_update Interface")
	{
		int value = 456;

		// Test elem_update (should work like push)
		REQUIRE(bf.elem_update(nullptr, &value, BPF_ANY) == 0);

		// Check if element might exist
		REQUIRE(bf.map_peek_elem(&value) == 0);

		// Test with non-null key (should be ignored)
		int dummy_key = 1;
		int value2 = 789;
		REQUIRE(bf.elem_update(&dummy_key, &value2, BPF_ANY) == 0);
		REQUIRE(bf.map_peek_elem(&value2) == 0);
	}

	SECTION("Invalid Operations")
	{
		int value = 123;

		// Test null value pointer
		errno = 0;
		REQUIRE(bf.map_push_elem(nullptr, BPF_ANY) == -1);
		REQUIRE(errno == EINVAL);

		errno = 0;
		REQUIRE(bf.map_peek_elem(nullptr) == -1);
		REQUIRE(errno == EINVAL);

		errno = 0;
		REQUIRE(bf.elem_update(nullptr, nullptr, BPF_ANY) == -1);
		REQUIRE(errno == EINVAL);

		// Test invalid flags
		errno = 0;
		REQUIRE(bf.map_push_elem(&value, BPF_EXIST) == -1);
		REQUIRE(errno == EINVAL);

		errno = 0;
		REQUIRE(bf.elem_update(nullptr, &value, BPF_EXIST) == -1);
		REQUIRE(errno == EINVAL);
	}

	SECTION("Unsupported Operations")
	{
		int value = 123;
		int next_key;

		// Test unsupported operations
		errno = 0;
		REQUIRE(bf.elem_delete(nullptr) == -1);
		REQUIRE(errno == EOPNOTSUPP);

		errno = 0;
		REQUIRE(bf.map_pop_elem(&value) == -1);
		REQUIRE(errno == EOPNOTSUPP);

		errno = 0;
		REQUIRE(bf.map_get_next_key(nullptr, &next_key) == -1);
		REQUIRE(errno == EOPNOTSUPP);

		// elem_lookup should return error (not supported)
		errno = 0;
		REQUIRE(bf.elem_lookup(nullptr) == nullptr);
		REQUIRE(errno == EOPNOTSUPP);

		errno = 0;
		int test_value = 999;
		REQUIRE(bf.elem_lookup(&test_value) == nullptr);
		REQUIRE(errno == EOPNOTSUPP);
	}
}

TEST_CASE("Bloom Filter False Positive Test", "[bloom_filter][false_positive]")
{
	const char *SHARED_MEMORY_NAME = "BloomFilterFPTestShm";
	const size_t SHARED_MEMORY_SIZE = 1024 * 1024;

	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	bloom_filter_map_impl bf(shm, sizeof(int), 10, 3); // Small filter for
							   // higher false
							   // positive rate

	// Add some known values
	std::vector<int> added_values = { 1, 2, 3, 4, 5 };
	for (int val : added_values) {
		REQUIRE(bf.map_push_elem(&val, BPF_ANY) == 0);
	}

	// Check that all added values are found
	for (int val : added_values) {
		REQUIRE(bf.map_peek_elem(&val) == 0);
	}

	// Test some values that were not added
	// Note: These might return 0 (false positive) or -1 (correctly not
	// found)
	std::vector<int> not_added_values = { 100, 200, 300, 400, 500 };
	int false_positives = 0;
	int true_negatives = 0;

	for (int val : not_added_values) {
		int result = bf.map_peek_elem(&val);
		if (result == 0) {
			false_positives++;
		} else {
			true_negatives++;
			REQUIRE(result == -1);
		}
	}

	// We should have at least some results (either false positives or true
	// negatives)
	REQUIRE((static_cast<size_t>(false_positives + true_negatives)) ==
		not_added_values.size());

	// Print some statistics for debugging
	INFO("False positives: " << false_positives << "/"
				 << not_added_values.size());
	INFO("True negatives: " << true_negatives << "/"
				<< not_added_values.size());
}

TEST_CASE("Bloom Filter Different Hash Counts", "[bloom_filter][hash_count]")
{
	const char *SHARED_MEMORY_NAME = "BloomFilterHashTestShm";
	const size_t SHARED_MEMORY_SIZE = 1024 * 1024;

	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	// Test different hash function counts
	for (unsigned int nr_hashes = 1; nr_hashes <= 10; nr_hashes++) {
		bloom_filter_map_impl bf(shm, sizeof(int), 50, nr_hashes);

		REQUIRE(bf.get_nr_hashes() == nr_hashes);

		// Add a test value
		int test_value = 42;
		REQUIRE(bf.map_push_elem(&test_value, BPF_ANY) == 0);

		// Should be able to find it
		REQUIRE(bf.map_peek_elem(&test_value) == 0);
	}
}

TEST_CASE("Bloom Filter Hash Algorithm Benchmark", "[bloom_filter][benchmark]")
{
	const char *SHARED_MEMORY_NAME = "BloomFilterBenchmarkShmCatch2";
	const size_t SHARED_MEMORY_SIZE = 1024 * 1024;

	// RAII for shared memory segment removal
	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	managed_shared_memory memory(create_only, SHARED_MEMORY_NAME,
				     SHARED_MEMORY_SIZE);

	const unsigned int value_size = 8;
	const unsigned int max_entries = 10000;
	const unsigned int nr_hashes = 5;
	const int num_operations = 50000;

	// Test data
	std::vector<uint64_t> test_values;
	test_values.reserve(num_operations);
	for (int i = 0; i < num_operations; i++) {
		test_values.push_back(static_cast<uint64_t>(i * 12345 + 67890));
	}

	// Benchmark DJB2 algorithm
	auto start_time = std::chrono::high_resolution_clock::now();
	{
		bloom_filter_map_impl djb2_filter(memory, value_size,
						  max_entries, nr_hashes,
						  BloomHashAlgorithm::DJB2);

		// Insert operations
		for (const auto &value : test_values) {
			djb2_filter.map_push_elem(&value, BPF_ANY);
		}

		// Lookup operations
		int found_count = 0;
		for (const auto &value : test_values) {
			if (djb2_filter.map_peek_elem(
				    const_cast<uint64_t *>(&value)) == 0) {
				found_count++;
			}
		}

		REQUIRE(found_count == num_operations); // All should be found
							// (no false negatives)
	}
	auto djb2_time = std::chrono::high_resolution_clock::now() - start_time;

	// Benchmark JHASH algorithm
	start_time = std::chrono::high_resolution_clock::now();
	{
		bloom_filter_map_impl jhash_filter(memory, value_size,
						   max_entries, nr_hashes,
						   BloomHashAlgorithm::JHASH);

		// Insert operations
		for (const auto &value : test_values) {
			jhash_filter.map_push_elem(&value, BPF_ANY);
		}

		// Lookup operations
		int found_count = 0;
		for (const auto &value : test_values) {
			if (jhash_filter.map_peek_elem(
				    const_cast<uint64_t *>(&value)) == 0) {
				found_count++;
			}
		}

		REQUIRE(found_count == num_operations); // All should be found
							// (no false negatives)
	}
	auto jhash_time =
		std::chrono::high_resolution_clock::now() - start_time;

	// Report results
	auto djb2_ms =
		std::chrono::duration_cast<std::chrono::milliseconds>(djb2_time)
			.count();
	auto jhash_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
				jhash_time)
				.count();

	INFO("DJB2 algorithm time: " << djb2_ms << " ms");
	INFO("JHASH algorithm time: " << jhash_ms << " ms");

	if (djb2_ms > 0 && jhash_ms > 0) {
		double speedup = static_cast<double>(djb2_ms) / jhash_ms;
		if (speedup > 1.0) {
			INFO("JHASH is " << speedup << "x faster than DJB2");
		} else {
			INFO("DJB2 is " << (1.0 / speedup)
					<< "x faster than JHASH");
		}
	}

	// Both algorithms should complete successfully
	REQUIRE(djb2_ms >= 0);
	REQUIRE(jhash_ms >= 0);
}

TEST_CASE("Bloom Filter Hash Distribution Quality",
	  "[bloom_filter][distribution]")
{
	const char *SHARED_MEMORY_NAME = "BloomFilterDistributionShmCatch2";
	const size_t SHARED_MEMORY_SIZE = 1024 * 1024;

	// RAII for shared memory segment removal
	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	managed_shared_memory memory(create_only, SHARED_MEMORY_NAME,
				     SHARED_MEMORY_SIZE);

	// Use more reasonable parameters:
	// - Set max_entries to the number of elements actually added
	// - Increase the number of hash functions
	// - Reduce test data volume to avoid overload
	const unsigned int value_size = 4;
	const unsigned int max_entries = 1000; // Designed capacity
	const unsigned int nr_hashes = 5; // More hash functions
	const int num_elements_to_add = 500; // Add half of capacity
	const int num_test_elements = 1000; // Number of test elements

	// Create better test data - use random distribution instead of
	// sequential integers
	std::vector<uint32_t> added_values;
	std::vector<uint32_t> test_values;

	// Generate values to add (use large gaps to avoid patterns)
	for (int i = 0; i < num_elements_to_add; i++) {
		added_values.push_back(static_cast<uint32_t>(i * 1000 + 12345));
	}

	// Generate test values (ensure no overlap with added values)
	for (int i = 0; i < num_test_elements; i++) {
		test_values.push_back(static_cast<uint32_t>(i * 1000 + 500000));
	}

	// Test false positive rates for both algorithms
	auto test_false_positive_rate = [&](BloomHashAlgorithm algo,
					    const char *algo_name) {
		bloom_filter_map_impl filter(memory, value_size, max_entries,
					     nr_hashes, algo);

		// Add predefined values
		for (const auto &value : added_values) {
			filter.map_push_elem(&value, BPF_ANY);
		}

		// Test values not in the filter
		int false_positives = 0;
		int true_negatives = 0;

		for (const auto &test_value : test_values) {
			uint32_t value = test_value;
			if (filter.map_peek_elem(&value) == 0) {
				false_positives++;
			} else {
				true_negatives++;
			}
		}

		double false_positive_rate =
			static_cast<double>(false_positives) /
			test_values.size();

		// Compute theoretical false positive rate
		// p = (1 - e^(-k*n/m))^k
		// where k=nr_hashes, n=added_values.size(), m=bit_array_size
		double k = static_cast<double>(nr_hashes);
		double n = static_cast<double>(added_values.size());
		double m = static_cast<double>(max_entries * nr_hashes * 7 / 5);
		// Adjust to power of two
		while (m < (max_entries * nr_hashes * 7 / 5)) {
			m *= 2;
		}
		double theoretical_fp_rate =
			std::pow(1.0 - std::exp(-k * n / m), k);

		INFO(algo_name << " false positive rate: "
			       << (false_positive_rate * 100) << "%");
		INFO(algo_name << " theoretical rate: "
			       << (theoretical_fp_rate * 100) << "%");
		INFO(algo_name << " false positives: " << false_positives
			       << ", true negatives: " << true_negatives);
		INFO(algo_name << " bit array size: " << static_cast<int>(m)
			       << " bits");

		// False positive rate should be within a reasonable range (<
		// 10%)
		REQUIRE(false_positive_rate < 0.1);

		// Actual false positive rate should not far exceed the
		// theoretical value (allow some deviation)
		REQUIRE(false_positive_rate < theoretical_fp_rate * 3.0);

		return false_positive_rate;
	};

	double djb2_fp_rate =
		test_false_positive_rate(BloomHashAlgorithm::DJB2, "DJB2");
	double jhash_fp_rate =
		test_false_positive_rate(BloomHashAlgorithm::JHASH, "JHASH");

	// Both should have reasonable false positive rates
	REQUIRE(djb2_fp_rate >= 0.0);
	REQUIRE(jhash_fp_rate >= 0.0);
	REQUIRE(djb2_fp_rate < 0.1);
	REQUIRE(jhash_fp_rate < 0.1);
}

TEST_CASE("Bloom Filter Large Scale False Positive Test",
	  "[bloom_filter][large_scale]")
{
	const char *SHARED_MEMORY_NAME = "BloomFilterLargeScaleShmCatch2";
	const size_t SHARED_MEMORY_SIZE = 4 * 1024 * 1024; // 4MB

	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	managed_shared_memory memory(create_only, SHARED_MEMORY_NAME,
				     SHARED_MEMORY_SIZE);

	// Large-scale test parameters
	const unsigned int value_size = 8;
	const unsigned int max_entries = 10000;
	const unsigned int nr_hashes = 5;
	const int num_elements_to_add = 8000; // 80% load
	const int num_test_elements = 50000; // Large test dataset

	INFO("Testing with " << num_elements_to_add << " added elements and "
			     << num_test_elements << " test elements");

	auto test_large_scale = [&](BloomHashAlgorithm algo,
				    const char *algo_name) {
		// Create a separate shared memory segment for each algorithm
		std::string shm_name =
			std::string(SHARED_MEMORY_NAME) + "_" + algo_name;

		shm_remove algo_remover((std::move(shm_name)));

		managed_shared_memory algo_memory(create_only, shm_name.c_str(),
						  SHARED_MEMORY_SIZE);

		bloom_filter_map_impl filter(algo_memory, value_size,
					     max_entries, nr_hashes, algo);

		// Generate random data instead of patterned data
		std::vector<uint64_t> added_values;
		std::vector<uint64_t> test_values;

		// Use a more sophisticated random number generator
		std::random_device rd;
		std::mt19937_64 gen(12345); // Fixed seed for reproducibility
		std::uniform_int_distribution<uint64_t> dis(1, UINT64_MAX);

		// Generate values to add
		std::set<uint64_t> added_set; // Ensure uniqueness
		while (added_set.size() <
		       static_cast<size_t>(num_elements_to_add)) {
			added_set.insert(dis(gen));
		}
		added_values.assign(added_set.begin(), added_set.end());

		// Generate test values (ensure no overlap with added values)
		std::set<uint64_t> test_set;
		while (test_set.size() <
		       static_cast<size_t>(num_test_elements)) {
			uint64_t val = dis(gen);
			if (added_set.find(val) == added_set.end()) {
				test_set.insert(val);
			}
		}
		test_values.assign(test_set.begin(), test_set.end());

		INFO("Generated " << added_values.size()
				  << " unique added values");
		INFO("Generated " << test_values.size()
				  << " unique test values");

		// Add elements to the bloom filter
		for (const auto &value : added_values) {
			REQUIRE(filter.map_push_elem(&value, BPF_ANY) == 0);
		}

		// Verify all added elements are found (no false negatives)
		int false_negatives = 0;
		for (const auto &value : added_values) {
			uint64_t val = value;
			if (filter.map_peek_elem(&val) != 0) {
				false_negatives++;
			}
		}
		REQUIRE(false_negatives == 0); // No false negatives expected

		// Test false positives
		int false_positives = 0;
		int true_negatives = 0;

		for (const auto &value : test_values) {
			uint64_t val = value;
			if (filter.map_peek_elem(&val) == 0) {
				false_positives++;
			} else {
				true_negatives++;
			}
		}

		double false_positive_rate =
			static_cast<double>(false_positives) /
			test_values.size();

		// Compute theoretical false positive rate
		double k = static_cast<double>(nr_hashes);
		double n = static_cast<double>(added_values.size());
		double m = static_cast<double>(max_entries * nr_hashes * 7 / 5);
		// Adjust to the next power of two
		size_t bit_array_size = 1;
		while (bit_array_size < static_cast<size_t>(m)) {
			bit_array_size <<= 1;
		}
		m = static_cast<double>(bit_array_size);

		double theoretical_fp_rate =
			std::pow(1.0 - std::exp(-k * n / m), k);

		INFO(algo_name << " large scale results:");
		INFO("  Added elements: " << added_values.size());
		INFO("  Test elements: " << test_values.size());
		INFO("  False positives: " << false_positives);
		INFO("  True negatives: " << true_negatives);
		INFO("  False positive rate: " << (false_positive_rate * 100)
					       << "%");
		INFO("  Theoretical rate: " << (theoretical_fp_rate * 100)
					    << "%");
		INFO("  Bit array size: " << static_cast<size_t>(m) << " bits");
		INFO("  Load factor: " << (n / max_entries * 100) << "%");

		// False positive rate should be reasonable
		REQUIRE(false_positive_rate >= 0.0);
		REQUIRE(false_positive_rate < 0.5); // Should not exceed 50%

		// If the theoretical rate is very low, actual rate may be 0,
		// which is fine
		if (theoretical_fp_rate > 0.001) {
			// Only check closeness when the theoretical value is
			// large enough
			REQUIRE(false_positive_rate <=
				theoretical_fp_rate * 5.0);
		}

		return false_positive_rate;
	};

	double djb2_fp_rate =
		test_large_scale(BloomHashAlgorithm::DJB2, "DJB2");
	double jhash_fp_rate =
		test_large_scale(BloomHashAlgorithm::JHASH, "JHASH");

	// Compare the two algorithms
	INFO("Algorithm comparison:");
	INFO("  DJB2 false positive rate: " << (djb2_fp_rate * 100) << "%");
	INFO("  JHASH false positive rate: " << (jhash_fp_rate * 100) << "%");
}

TEST_CASE("Bloom Filter Extreme Load Test", "[bloom_filter][extreme]")
{
	const char *SHARED_MEMORY_NAME = "BloomFilterExtremeShmCatch2";
	const size_t SHARED_MEMORY_SIZE = 8 * 1024 * 1024; // 8MB

	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	managed_shared_memory memory(create_only, SHARED_MEMORY_NAME,
				     SHARED_MEMORY_SIZE);

	// Extreme test parameters - overloaded bloom filter
	const unsigned int value_size = 8;
	const unsigned int max_entries = 5000; // Smaller designed capacity
	const unsigned int nr_hashes = 3; // Fewer hash functions
	const int num_elements_to_add = 8000; // 160% overloaded
	const int num_test_elements = 100000; // Large dataset

	INFO("EXTREME TEST: "
	     << num_elements_to_add << " elements in filter designed for "
	     << max_entries << " (load factor: "
	     << (num_elements_to_add * 100 / max_entries) << "%)");

	auto test_extreme_load = [&](BloomHashAlgorithm algo,
				     const char *algo_name) {
		// Create a dedicated shared memory segment for each algorithm
		std::string shm_name =
			std::string(SHARED_MEMORY_NAME) + "_" + algo_name;

		shm_remove algo_remover((std::move(shm_name)));

		managed_shared_memory algo_memory(create_only, shm_name.c_str(),
						  SHARED_MEMORY_SIZE);

		bloom_filter_map_impl filter(algo_memory, value_size,
					     max_entries, nr_hashes, algo);

		// Generate random data
		std::vector<uint64_t> added_values;
		std::vector<uint64_t> test_values;

		std::mt19937_64 gen(54321); // Different seed
		std::uniform_int_distribution<uint64_t> dis(1, UINT64_MAX);

		// Generate values to add
		std::set<uint64_t> added_set;
		while (added_set.size() <
		       static_cast<size_t>(num_elements_to_add)) {
			added_set.insert(dis(gen));
		}
		added_values.assign(added_set.begin(), added_set.end());

		// Generate test values
		std::set<uint64_t> test_set;
		while (test_set.size() <
		       static_cast<size_t>(num_test_elements)) {
			uint64_t val = dis(gen);
			if (added_set.find(val) == added_set.end()) {
				test_set.insert(val);
			}
		}
		test_values.assign(test_set.begin(), test_set.end());

		// Add elements to the bloom filter
		for (const auto &value : added_values) {
			REQUIRE(filter.map_push_elem(&value, BPF_ANY) == 0);
		}

		// Verify all added elements are found
		int false_negatives = 0;
		for (const auto &value : added_values) {
			uint64_t val = value;
			if (filter.map_peek_elem(&val) != 0) {
				false_negatives++;
			}
		}
		REQUIRE(false_negatives == 0);

		// Test false positives
		int false_positives = 0;
		int true_negatives = 0;

		for (const auto &value : test_values) {
			uint64_t val = value;
			if (filter.map_peek_elem(&val) == 0) {
				false_positives++;
			} else {
				true_negatives++;
			}
		}

		double false_positive_rate =
			static_cast<double>(false_positives) /
			test_values.size();

		// Compute theoretical false positive rate
		double k = static_cast<double>(nr_hashes);
		double n = static_cast<double>(added_values.size());
		double m = static_cast<double>(max_entries * nr_hashes * 7 / 5);
		size_t bit_array_size = 1;
		while (bit_array_size < static_cast<size_t>(m)) {
			bit_array_size <<= 1;
		}
		m = static_cast<double>(bit_array_size);

		double theoretical_fp_rate =
			std::pow(1.0 - std::exp(-k * n / m), k);

		INFO(algo_name << " EXTREME load results:");
		INFO("  Added elements: " << added_values.size());
		INFO("  Test elements: " << test_values.size());
		INFO("  False positives: " << false_positives);
		INFO("  True negatives: " << true_negatives);
		INFO("  False positive rate: " << (false_positive_rate * 100)
					       << "%");
		INFO("  Theoretical rate: " << (theoretical_fp_rate * 100)
					    << "%");
		INFO("  Bit array size: " << static_cast<size_t>(m) << " bits");
		INFO("  Load factor: " << (n / max_entries * 100) << "%");

		// Under extreme load, false positive rate should be high but
		// still reasonable
		REQUIRE(false_positive_rate >= 0.0);
		REQUIRE(false_positive_rate < 1.0); // Should not be 100%

		// Under overload, actual rate may far exceed theoretical value,
		// which is expected
		INFO("  Ratio to theoretical: "
		     << (false_positive_rate / theoretical_fp_rate));

		return false_positive_rate;
	};

	double djb2_fp_rate =
		test_extreme_load(BloomHashAlgorithm::DJB2, "DJB2");
	double jhash_fp_rate =
		test_extreme_load(BloomHashAlgorithm::JHASH, "JHASH");

	// Compare both algorithms under extreme load
	INFO("EXTREME load algorithm comparison:");
	INFO("  DJB2 false positive rate: " << (djb2_fp_rate * 100) << "%");
	INFO("  JHASH false positive rate: " << (jhash_fp_rate * 100) << "%");

	// Both algorithms should have relatively high false positive rates
	REQUIRE(djb2_fp_rate > 0.01); // At least 1%
	REQUIRE(jhash_fp_rate > 0.01); // At least 1%
}