#include "catch2/catch_test_macros.hpp"
#include "catch2/catch_message.hpp" // For INFO, FAIL

#include "bpf_map/userspace/stack.hpp" // Adjust path as needed
#include "../common_def.hpp"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <vector>
#include <thread>
#include <chrono> // For std::chrono::milliseconds
#include <cstring> // For strcmp, strerror
#include <cerrno> // For errno, ENOENT, E2BIG, EINVAL, ENOTSUP
#include <pthread.h> // For pthread_spinlock_t

// Define a helper structure for testing
struct TestValue {
	int id;
	char data[20];

	bool operator==(const TestValue &other) const
	{
		return id == other.id && strcmp(data, other.data) == 0;
	}
	// Add a stream insertion operator for Catch2 to print TestValue on
	// failure
	friend std::ostream &operator<<(std::ostream &os, const TestValue &tv)
	{
		os << "TestValue{id=" << tv.id << ", data=\"" << tv.data
		   << "\"}";
		return os;
	}
};

TEST_CASE("Stack Map Constructor Validation", "[stack_map][constructor]")
{
	const char *SHARED_MEMORY_NAME = "StackMapConstructorTestShmCatch2";
	const size_t SHARED_MEMORY_SIZE = 1024; // Small, just for
						// constructor test

	// RAII for shared memory segment removal
	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	// Test invalid max_entries (0)
	REQUIRE_THROWS_AS(bpftime::stack_map_impl(shm, 0, 0),
			  std::invalid_argument);

	// Test invalid value_size (0)
	REQUIRE_THROWS_AS(bpftime::stack_map_impl(shm, 0, 5),
			  std::invalid_argument);
}

TEST_CASE("Stack Map Core Operations", "[stack_map][core]")
{
	const char *SHARED_MEMORY_NAME = "StackMapCoreTestShmCatch2";
	const size_t SHARED_MEMORY_SIZE = 65536; // 64KB

	// RAII for shared memory segment removal
	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	bpftime::stack_map_impl *s_map_int = nullptr;
	bpftime::stack_map_impl *s_map_struct = nullptr;

	try {
		// Construct maps for integers
		s_map_int = shm.construct<bpftime::stack_map_impl>(
			"StackMapIntInstance")(shm, sizeof(int), 3);
		REQUIRE(s_map_int != nullptr);

		// Construct maps for structures
		s_map_struct = shm.construct<bpftime::stack_map_impl>(
			"StackMapStructInstance")(shm, sizeof(TestValue), 2);
		REQUIRE(s_map_struct != nullptr);
	} catch (const std::exception &ex) {
		FAIL("Failed during map construction in shared memory: "
		     << ex.what());
	}

	SECTION("Basic Int Operations - LIFO Behavior")
	{
		REQUIRE(s_map_int != nullptr); // Ensure map was created
		int val1 = 10, val2 = 20, val3 = 30, val4 = 40;
		int peek_val, pop_val;

		// 1. Peek/Pop empty stack
		errno = 0;
		REQUIRE(s_map_int->map_peek_elem(&peek_val) == -1);
		REQUIRE(errno == ENOENT);

		errno = 0;
		REQUIRE(s_map_int->map_pop_elem(&pop_val) == -1);
		REQUIRE(errno == ENOENT);

		// 2. Push three elements (stack capacity is 3)
		REQUIRE(s_map_int->map_push_elem(&val1, BPF_ANY) == 0); // [10]
		REQUIRE(s_map_int->map_push_elem(&val2, BPF_ANY) == 0); // [10,
									// 20]
		REQUIRE(s_map_int->map_push_elem(&val3, BPF_ANY) == 0); // [10,
									// 20,
									// 30]

		// 3. Stack is full, try BPF_ANY Push
		errno = 0;
		REQUIRE(s_map_int->map_push_elem(&val4, BPF_ANY) == -1);
		REQUIRE(errno == E2BIG);

		// 4. Peek top element (LIFO - should be last pushed: 30)
		REQUIRE(s_map_int->map_peek_elem(&peek_val) == 0);
		REQUIRE(peek_val == val3); // Should be 30 (top of stack)

		// 5. Pop top element (LIFO - should be 30)
		REQUIRE(s_map_int->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == val3); // Should be 30

		// 6. Peek current top element (should be 20)
		REQUIRE(s_map_int->map_peek_elem(&peek_val) == 0);
		REQUIRE(peek_val == val2); // Should be 20

		// 7. Push a new element (stack has space: [10, 20, _])
		REQUIRE(s_map_int->map_push_elem(&val4, BPF_ANY) == 0); // [10,
									// 20,
									// 40]

		// 8. Pop remaining elements, verify LIFO order: 40, 20, 10
		REQUIRE(s_map_int->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == val4); // 40 (last pushed)
		REQUIRE(s_map_int->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == val2); // 20
		REQUIRE(s_map_int->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == val1); // 10 (first pushed)

		// 9. Stack is now empty
		errno = 0;
		REQUIRE(s_map_int->map_pop_elem(&pop_val) == -1);
		REQUIRE(errno == ENOENT);
	}

	SECTION("Push With Exist Flag (Struct Operations)")
	{
		REQUIRE(s_map_struct != nullptr); // Ensure map was created
		TestValue tv1 = { 1, "First" };
		TestValue tv2 = { 2, "Second" };
		TestValue tv3 = { 3, "Third" };
		TestValue peek_val, pop_val;

		// s_map_struct capacity is 2

		// 1. Push two elements to fill the stack
		REQUIRE(s_map_struct->map_push_elem(&tv1, BPF_ANY) ==
			0); // [tv1]
		REQUIRE(s_map_struct->map_push_elem(&tv2, BPF_ANY) ==
			0); // [tv1, tv2]

		// 2. Stack is full, try BPF_ANY Push (should fail)
		errno = 0;
		REQUIRE(s_map_struct->map_push_elem(&tv3, BPF_ANY) == -1);
		REQUIRE(errno == E2BIG);

		// 3. Stack is full, try BPF_EXIST Push (should succeed,
		// removing oldest)
		REQUIRE(s_map_struct->map_push_elem(&tv3, BPF_EXIST) ==
			0); // [tv2, tv3]

		// 4. Peek to verify the top element is tv3 (LIFO)
		REQUIRE(s_map_struct->map_peek_elem(&peek_val) == 0);
		REQUIRE(peek_val == tv3);

		// 5. Pop to verify LIFO order: tv3, tv2 (tv1 was removed)
		REQUIRE(s_map_struct->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == tv3); // tv3 is popped (top)

		REQUIRE(s_map_struct->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == tv2); // tv2 is popped

		// 6. Stack is empty
		errno = 0;
		REQUIRE(s_map_struct->map_pop_elem(&pop_val) == -1);
		REQUIRE(errno == ENOENT);
	}

	SECTION("Invalid Push Flags")
	{
		REQUIRE(s_map_int != nullptr);
		int val = 100;

		// BPF_NOEXIST (1) is an invalid push flag
		errno = 0;
		REQUIRE(s_map_int->map_push_elem(&val, BPF_NOEXIST) == -1);
		REQUIRE(errno == EINVAL);

		// Other invalid flags (e.g., 4)
		errno = 0;
		REQUIRE(s_map_int->map_push_elem(&val, 4) == -1);
		REQUIRE(errno == EINVAL);
	}

	SECTION("Null Pointer Handling")
	{
		REQUIRE(s_map_int != nullptr);
		int val = 100;

		// Test null value pointer in push
		errno = 0;
		REQUIRE(s_map_int->map_push_elem(nullptr, BPF_ANY) == -1);
		REQUIRE(errno == EINVAL);

		// Test null value pointer in pop
		errno = 0;
		REQUIRE(s_map_int->map_pop_elem(nullptr) == -1);
		REQUIRE(errno == EINVAL);

		// Test null value pointer in peek
		errno = 0;
		REQUIRE(s_map_int->map_peek_elem(nullptr) == -1);
		REQUIRE(errno == EINVAL);
	}

	SECTION("Standard Map Interface Behavior (Stack Operations)")
	{
		REQUIRE(s_map_int != nullptr);
		int key = 0; // Stack is keyless, but API needs a param
		int val = 100;
		int next_key;

		// elem_lookup should work as peek operation on empty stack
		errno = 0;
		REQUIRE(s_map_int->elem_lookup(&key) == nullptr);
		REQUIRE(errno == ENOENT); // Check errno was set by elem_lookup
					  // for empty stack

		// elem_update should work as push operation
		REQUIRE(s_map_int->elem_update(&key, &val, BPF_ANY) == 0);

		// Now elem_lookup should return a valid pointer
		void *ptr = s_map_int->elem_lookup(&key);
		REQUIRE(ptr != nullptr);
		REQUIRE(*(int *)ptr == val);

		// elem_delete should work as pop operation
		REQUIRE(s_map_int->elem_delete(&key) == 0);

		// Stack should be empty again
		errno = 0;
		REQUIRE(s_map_int->elem_lookup(&key) == nullptr);
		REQUIRE(errno == ENOENT);

		errno = 0;
		REQUIRE(s_map_int->elem_delete(&key) == -1);
		REQUIRE(errno == ENOENT);

		// map_get_next_key is not supported for stacks
		errno = 0;
		REQUIRE(s_map_int->map_get_next_key(&key, &next_key) == -1);
		REQUIRE(errno == EINVAL);

		// Test key as nullptr for map_get_next_key
		errno = 0;
		REQUIRE(s_map_int->map_get_next_key(nullptr, &next_key) == -1);
		REQUIRE(errno == EINVAL);
	}

	SECTION("LIFO vs FIFO Comparison Test")
	{
		REQUIRE(s_map_int != nullptr);

		// Push elements in order: 1, 2, 3
		int vals[] = { 1, 2, 3 };
		for (int i = 0; i < 3; i++) {
			REQUIRE(s_map_int->map_push_elem(&vals[i], BPF_ANY) ==
				0);
		}

		// Pop elements - should come out in LIFO order: 3, 2, 1
		int pop_val;
		REQUIRE(s_map_int->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == 3); // Last in, first out

		REQUIRE(s_map_int->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == 2);

		REQUIRE(s_map_int->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == 1); // First in, last out
	}

	// Cleanup objects constructed in shared memory
	if (s_map_int) {
		shm.destroy_ptr(s_map_int);
		s_map_int = nullptr;
	}
	if (s_map_struct) {
		shm.destroy_ptr(s_map_struct);
		s_map_struct = nullptr;
	}
}

TEST_CASE("Stack Map Conceptual Concurrency Test",
	  "[stack_map][concurrency][.disabled]")
{
	const char *SHARED_MEMORY_NAME_CONC = "StackMapConcTestShmCatch2";
	const size_t SHARED_MEMORY_SIZE_CONC = 65536;

	shm_remove remover((std::string(SHARED_MEMORY_NAME_CONC)));

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME_CONC,
		SHARED_MEMORY_SIZE_CONC);

	bpftime::stack_map_impl *s_map_conc = nullptr;
	pthread_spinlock_t map_lock; // Add spinlock for thread safety
	pthread_spin_init(&map_lock, 0);

	try {
		s_map_conc = shm.construct<bpftime::stack_map_impl>(
			"StackMapConcInstance")(shm, sizeof(int),
						10); // Larger capacity
		REQUIRE(s_map_conc != nullptr);
	} catch (const std::exception &ex) {
		FAIL("Failed to construct s_map_conc: " << ex.what());
	}

	const int num_threads = 5;
	const int ops_per_thread = 20;
	std::vector<std::thread> threads;

	auto thread_func = [&](int thread_id) {
		for (int i = 0; i < ops_per_thread; ++i) {
			int val_to_push = thread_id * 1000 + i;
			// Using BPF_EXIST to avoid simple gridlock on full
			{
				pthread_spin_lock(&map_lock);
				s_map_conc->map_push_elem(&val_to_push,
							  BPF_EXIST);
				pthread_spin_unlock(&map_lock);
			}

			// Brief sleep to increase chance of thread interleaving
			std::this_thread::sleep_for(
				std::chrono::microseconds(100));

			int popped_val;
			// Try to pop, ignore ENOENT as other threads might
			// empty it
			{
				pthread_spin_lock(&map_lock);
				s_map_conc->map_pop_elem(&popped_val);
				pthread_spin_unlock(&map_lock);
			}
		}
	};

	for (int i = 0; i < num_threads; ++i) {
		threads.emplace_back(thread_func, i);
	}

	for (auto &t : threads) {
		if (t.joinable()) {
			t.join();
		}
	}

	// Empty the stack and verify it becomes empty
	int final_pop_val;
	int pop_count = 0;
	while (true) {
		pthread_spin_lock(&map_lock);
		int result = s_map_conc->map_pop_elem(&final_pop_val);
		pthread_spin_unlock(&map_lock);
		if (result == 0) {
			pop_count++;
		} else {
			break;
		}
	}
	INFO("Final pop count from concurrent stack: " << pop_count);

	{
		pthread_spin_lock(&map_lock);
		errno = 0;
		REQUIRE(s_map_conc->map_pop_elem(&final_pop_val) == -1);
		REQUIRE(errno == ENOENT); // Must be empty eventually
		pthread_spin_unlock(&map_lock);
	}

	if (s_map_conc) {
		shm.destroy_ptr(s_map_conc);
	}

	pthread_spin_destroy(&map_lock);
}

// Kernel-style tests using bpftime internal API
// Based on kernel test_stackmap function
TEST_CASE("Stack Map Kernel Style Test", "[stack_map][kernel_style]")
{
	const char *SHARED_MEMORY_NAME = "StackMapKernelTestShmCatch2";
	const size_t SHARED_MEMORY_SIZE = 65536;
	const int MAP_SIZE = 32;

	// RAII for shared memory segment removal
	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	// Fill test values to be used
	uint32_t vals[MAP_SIZE + MAP_SIZE / 2];
	for (int i = 0; i < MAP_SIZE + MAP_SIZE / 2; i++)
		vals[i] = rand();

	bpftime::stack_map_impl *s_map = nullptr;

	// Test invalid value_size (should fail in constructor)
	REQUIRE_THROWS_AS(bpftime::stack_map_impl(shm, 0, MAP_SIZE),
			  std::invalid_argument);

	// Create valid stack map
	try {
		s_map = shm.construct<bpftime::stack_map_impl>(
			"StackMapKernelInstance")(shm, sizeof(uint32_t),
						  MAP_SIZE);
		REQUIRE(s_map != nullptr);
	} catch (const std::exception &ex) {
		FAIL("Failed during map construction: " << ex.what());
	}

	// Push MAP_SIZE elements
	for (int i = 0; i < MAP_SIZE; i++) {
		REQUIRE(s_map->map_push_elem(&vals[i], BPF_ANY) == 0);
	}

	// Check that element cannot be pushed due to max_entries limit
	uint32_t val;
	errno = 0;
	REQUIRE(s_map->map_push_elem(&val, BPF_ANY) == -1);
	REQUIRE(errno == E2BIG);

	// Peek element (should be last element - LIFO)
	REQUIRE(s_map->map_peek_elem(&val) == 0);
	REQUIRE(val == vals[MAP_SIZE - 1]); // Last pushed element

	// Replace half elements using BPF_EXIST
	for (int i = MAP_SIZE; i < MAP_SIZE + MAP_SIZE / 2; i++) {
		REQUIRE(s_map->map_push_elem(&vals[i], BPF_EXIST) == 0);
	}

	// Pop all elements - should get them in LIFO order
	for (int i = MAP_SIZE + MAP_SIZE / 2 - 1; i >= MAP_SIZE / 2; i--) {
		REQUIRE(s_map->map_pop_elem(&val) == 0);
		REQUIRE(val == vals[i]);
	}

	// Check that there are no elements left
	errno = 0;
	REQUIRE(s_map->map_pop_elem(&val) == -1);
	REQUIRE(errno == ENOENT);

	// Check that non-supported functions set errno to EINVAL
	// Note: elem_delete with nullptr returns ENOENT, but we want to test
	// unsupported operations Let's test with map_get_next_key instead as
	// that's truly unsupported
	errno = 0;
	REQUIRE(s_map->elem_delete(&val) == -1); // Try to delete from empty
						 // stack
	REQUIRE(errno == ENOENT);

	uint32_t next_key;
	errno = 0;
	REQUIRE(s_map->map_get_next_key(nullptr, &next_key) == -1);
	REQUIRE(errno == EINVAL);

	// Cleanup
	if (s_map) {
		shm.destroy_ptr(s_map);
	}
}

// Test using standard map interface (elem_update, elem_lookup, etc.)
TEST_CASE("Stack Map Standard Interface Kernel Style",
	  "[stack_map][kernel_style][std_interface]")
{
	const char *SHARED_MEMORY_NAME = "StackMapStdKernelTestShmCatch2";
	const size_t SHARED_MEMORY_SIZE = 65536;
	const int MAP_SIZE = 32;

	// RAII for shared memory segment removal
	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	// Fill test values to be used
	uint32_t vals[MAP_SIZE + MAP_SIZE / 2];
	for (int i = 0; i < MAP_SIZE + MAP_SIZE / 2; i++)
		vals[i] = rand();

	bpftime::stack_map_impl *s_map = nullptr;

	try {
		s_map = shm.construct<bpftime::stack_map_impl>(
			"StackMapStdInstance")(shm, sizeof(uint32_t), MAP_SIZE);
		REQUIRE(s_map != nullptr);
	} catch (const std::exception &ex) {
		FAIL("Failed during map construction: " << ex.what());
	}

	int key = 0; // Stack is keyless, but standard interface needs a key

	// Push MAP_SIZE elements using elem_update
	for (int i = 0; i < MAP_SIZE; i++) {
		REQUIRE(s_map->elem_update(&key, &vals[i], BPF_ANY) == 0);
	}

	// Check that element cannot be pushed due to max_entries limit
	uint32_t val;
	errno = 0;
	REQUIRE(s_map->elem_update(&key, &val, BPF_ANY) == -1);
	REQUIRE(errno == E2BIG);

	// Peek element using elem_lookup (should be last element - LIFO)
	void *ptr = s_map->elem_lookup(&key);
	REQUIRE(ptr != nullptr);
	REQUIRE(*(uint32_t *)ptr == vals[MAP_SIZE - 1]);

	// Replace half elements using elem_update with BPF_EXIST
	for (int i = MAP_SIZE; i < MAP_SIZE + MAP_SIZE / 2; i++) {
		REQUIRE(s_map->elem_update(&key, &vals[i], BPF_EXIST) == 0);
	}

	// Pop all elements using elem_delete
	for (int i = MAP_SIZE + MAP_SIZE / 2 - 1; i >= MAP_SIZE / 2; i--) {
		REQUIRE(s_map->elem_delete(&key) == 0);
		// Note: elem_delete doesn't return the value, so we can't
		// verify the exact value
	}

	// Check that there are no elements left
	errno = 0;
	REQUIRE(s_map->elem_lookup(&key) == nullptr);
	REQUIRE(errno == ENOENT);

	errno = 0;
	REQUIRE(s_map->elem_delete(&key) == -1);
	REQUIRE(errno == ENOENT);

	// Check that non-supported functions set errno to EINVAL
	uint32_t next_key;
	errno = 0;
	REQUIRE(s_map->map_get_next_key(&key, &next_key) == -1);
	REQUIRE(errno == EINVAL);

	errno = 0;
	REQUIRE(s_map->map_get_next_key(nullptr, &next_key) == -1);
	REQUIRE(errno == EINVAL);

	// Cleanup
	if (s_map) {
		shm.destroy_ptr(s_map);
	}
}