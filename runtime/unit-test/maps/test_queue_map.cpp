#include "catch2/catch_test_macros.hpp"
#include "catch2/catch_message.hpp" // For INFO, FAIL
// It's good practice to include specific headers if catch_all.hpp is not used.
// For REQUIRE_THROWS_AS, catch_test_macros.hpp is usually sufficient.

#include "bpf_map/userspace/queue_map.hpp" // Adjust path as needed
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <vector>
#include <thread>
#include <chrono> // For std::chrono::milliseconds
#include <cstring> // For strcmp, strerror
#include <cerrno>  // For errno, ENOENT, E2BIG, EINVAL, ENOTSUP

// Define a helper structure, same as in GTest version
struct TestValue {
	int id;
	char data[20];

	bool operator==(const TestValue &other) const
	{
		return id == other.id && strcmp(data, other.data) == 0;
	}
	// Add a stream insertion operator for Catch2 to print TestValue on failure
	friend std::ostream& operator<<(std::ostream& os, const TestValue& tv) {
		os << "TestValue{id=" << tv.id << ", data=\"" << tv.data << "\"}";
		return os;
	}
};

// Use Catch2's TEST_CASE and SECTION macros
// Shared memory setup will be done per TEST_CASE or using RAII helpers.

TEST_CASE("Queue Map Constructor Validation", "[queue_map][constructor]")
{
	const char *SHARED_MEMORY_NAME_CV = "QueueMapCVTestShmCatch2";
	const size_t SHARED_MEMORY_SIZE_CV = 1024; // Small, just for constructor test

	// RAII for shared memory segment removal
	struct ShmRemover {
		const char *name;
		ShmRemover(const char *n) : name(n) {
			// Pre-cleanup
			boost::interprocess::shared_memory_object::remove(name);
		}
		~ShmRemover() {
			// Post-cleanup
			boost::interprocess::shared_memory_object::remove(name);
		}
	} remover(SHARED_MEMORY_NAME_CV);

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME_CV,
		SHARED_MEMORY_SIZE_CV);

	// Test invalid key_size
	REQUIRE_THROWS_AS(bpftime::queue_map_impl(shm, 4, sizeof(int), 5,
						  bpftime::BPF_MAP_TYPE_QUEUE),
			  std::invalid_argument);

	// Test invalid max_entries (0)
	REQUIRE_THROWS_AS(bpftime::queue_map_impl(shm, 0, sizeof(int), 0,
						  bpftime::BPF_MAP_TYPE_QUEUE),
			  std::invalid_argument);

	// Test invalid value_size (0)
	REQUIRE_THROWS_AS(bpftime::queue_map_impl(shm, 0, 0, 5,
						  bpftime::BPF_MAP_TYPE_QUEUE),
			  std::invalid_argument);
}

TEST_CASE("Queue Map Core Operations", "[queue_map][core]")
{
	const char *SHARED_MEMORY_NAME = "QueueMapCoreTestShmCatch2";
	const size_t SHARED_MEMORY_SIZE = 65536; // 64KB

	// RAII for shared memory segment removal
	struct ShmRemover {
		const char *name;
		ShmRemover(const char *n) : name(n) {
			boost::interprocess::shared_memory_object::remove(name);
		}
		~ShmRemover() {
			boost::interprocess::shared_memory_object::remove(name);
		}
	} remover(SHARED_MEMORY_NAME);

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	bpftime::queue_map_impl *q_map_int = nullptr;
	bpftime::queue_map_impl *q_map_struct = nullptr;

	try {
		// Construct maps for integers
		q_map_int = shm.construct<bpftime::queue_map_impl>(
			"QueueMapIntInstance")(shm, 0, sizeof(int), 3,
					       bpftime::BPF_MAP_TYPE_QUEUE);
		REQUIRE(q_map_int != nullptr);

		// Construct maps for structures
		q_map_struct = shm.construct<bpftime::queue_map_impl>(
			"QueueMapStructInstance")(shm, 0, sizeof(TestValue), 2,
						  bpftime::BPF_MAP_TYPE_QUEUE);
		REQUIRE(q_map_struct != nullptr);
	} catch (const std::exception &ex) {
		FAIL("Failed during map construction in shared memory: "
		     << ex.what());
	}

	SECTION("Basic Int Operations")
	{
		REQUIRE(q_map_int != nullptr); // Ensure map was created
		int val1 = 10, val2 = 20, val3 = 30, val4 = 40;
		int peek_val, pop_val;

		// 1. Peek/Pop empty queue
		REQUIRE(q_map_int->map_peek_elem(&peek_val) == -ENOENT);
		REQUIRE(q_map_int->map_pop_elem(&pop_val) == -ENOENT);

		// 2. Push three elements (queue capacity is 3)
		REQUIRE(q_map_int->map_push_elem(&val1, BPF_ANY) == 0); // 10
		REQUIRE(q_map_int->map_push_elem(&val2, BPF_ANY) == 0); // 10, 20
		REQUIRE(q_map_int->map_push_elem(&val3, BPF_ANY) == 0); // 10, 20, 30

		// 3. Queue is full, try BPF_ANY Push
		REQUIRE(q_map_int->map_push_elem(&val4, BPF_ANY) == -E2BIG);

		// 4. Peek first element
		REQUIRE(q_map_int->map_peek_elem(&peek_val) == 0);
		REQUIRE(peek_val == val1); // Should be 10

		// 5. Pop first element
		REQUIRE(q_map_int->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == val1); // Should be 10

		// 6. Peek current first element
		REQUIRE(q_map_int->map_peek_elem(&peek_val) == 0);
		REQUIRE(peek_val == val2); // Should be 20

		// 7. Push a new element (queue has space: 20, 30, _)
		REQUIRE(q_map_int->map_push_elem(&val4, BPF_ANY) == 0); // 20, 30, 40

		// 8. Pop remaining elements, verify order
		REQUIRE(q_map_int->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == val2); // 20
		REQUIRE(q_map_int->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == val3); // 30
		REQUIRE(q_map_int->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == val4); // 40

		// 9. Queue is now empty
		REQUIRE(q_map_int->map_pop_elem(&pop_val) == -ENOENT);
	}

	SECTION("Push With Exist Flag (Struct Operations)")
	{
		REQUIRE(q_map_struct != nullptr); // Ensure map was created
		TestValue tv1 = { 1, "Hello" };
		TestValue tv2 = { 2, "World" };
		TestValue tv3 = { 3, "Overwrite" };
		TestValue peek_val, pop_val;

		// q_map_struct capacity is 2

		// 1. Push two elements to fill the queue
		REQUIRE(q_map_struct->map_push_elem(&tv1, BPF_ANY) == 0); // tv1
		REQUIRE(q_map_struct->map_push_elem(&tv2, BPF_ANY) == 0); // tv1, tv2

		// 2. Queue is full, try BPF_ANY Push (should fail)
		REQUIRE(q_map_struct->map_push_elem(&tv3, BPF_ANY) ==
			-E2BIG);

		// 3. Queue is full, try BPF_EXIST Push (should succeed, overwriting tv1)
		REQUIRE(q_map_struct->map_push_elem(&tv3, BPF_EXIST) == 0); // tv2, tv3

		// 4. Peek to verify the first element is tv2
		REQUIRE(q_map_struct->map_peek_elem(&peek_val) == 0);
		REQUIRE(peek_val == tv2);

		// 5. Pop to verify order
		REQUIRE(q_map_struct->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == tv2); // tv2 is popped

		REQUIRE(q_map_struct->map_pop_elem(&pop_val) == 0);
		REQUIRE(pop_val == tv3); // tv3 is popped

		// 6. Queue is empty
		REQUIRE(q_map_struct->map_pop_elem(&pop_val) == -ENOENT);
	}

	SECTION("Invalid Push Flags")
	{
		REQUIRE(q_map_int != nullptr);
		int val = 100;
		// BPF_NOEXIST (1) is an invalid push flag
		REQUIRE(q_map_int->map_push_elem(&val, BPF_NOEXIST) == -EINVAL);
		// Other invalid flags (e.g., 4)
		REQUIRE(q_map_int->map_push_elem(&val, 4) == -EINVAL);
	}

	SECTION("Standard Map Interface Behavior (Error Checks)")
	{
		REQUIRE(q_map_int != nullptr);
		int key = 0; // Queue is keyless, but API needs a param
		int val = 1;
		int next_key;

		// These generic operations should be unsupported for queue type
		errno = 0; // Clear errno before call
		REQUIRE(q_map_int->elem_lookup(&key) == nullptr);
		REQUIRE(errno == ENOTSUP); // Check errno was set by elem_lookup

		REQUIRE(q_map_int->elem_update(&key, &val, BPF_ANY) ==
			-ENOTSUP);
		REQUIRE(q_map_int->elem_delete(&key) == -ENOTSUP);
		
		REQUIRE(q_map_int->map_get_next_key(&key, &next_key) ==
			-EINVAL);
		// Test key as nullptr for map_get_next_key
		REQUIRE(q_map_int->map_get_next_key(nullptr, &next_key) ==
			-EINVAL);
	}

	// Cleanup objects constructed in shared memory
	// This is important before the shared memory segment itself is destroyed/unmapped.
	if (q_map_int) {
		shm.destroy_ptr(q_map_int);
		q_map_int = nullptr;
	}
	if (q_map_struct) {
		shm.destroy_ptr(q_map_struct);
		q_map_struct = nullptr;
	}
	// The ShmRemover RAII object will call shared_memory_object::remove at scope exit.
}

// Using [.disabled] or [!hide] tag to disable a test in Catch2
TEST_CASE("Queue Map Conceptual Concurrency Test", "[queue_map][concurrency][.disabled]")
{
	const char *SHARED_MEMORY_NAME_CONC = "QueueMapConcTestShmCatch2";
	const size_t SHARED_MEMORY_SIZE_CONC = 65536;

	struct ShmRemover {
		const char *name;
		ShmRemover(const char *n) : name(n) { boost::interprocess::shared_memory_object::remove(name); }
		~ShmRemover() { boost::interprocess::shared_memory_object::remove(name); }
	} remover(SHARED_MEMORY_NAME_CONC);

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME_CONC,
		SHARED_MEMORY_SIZE_CONC);
	
	bpftime::queue_map_impl *q_map_conc = nullptr;
	try {
		q_map_conc = shm.construct<bpftime::queue_map_impl>(
			"QueueMapConcInstance")(shm, 0, sizeof(int), 10, // Larger capacity for concurrency
					       bpftime::BPF_MAP_TYPE_QUEUE);
		REQUIRE(q_map_conc != nullptr);
	} catch (const std::exception &ex) {
		FAIL("Failed to construct q_map_conc: " << ex.what());
	}


	const int num_threads = 5;
	const int ops_per_thread = 20; // Increased ops
	std::vector<std::thread> threads;

	auto thread_func = [&](int thread_id) {
		for (int i = 0; i < ops_per_thread; ++i) {
			int val_to_push = thread_id * 1000 + i;
			// Using BPF_EXIST to avoid simple gridlock on full,
			// focusing on race conditions rather than capacity issues.
			q_map_conc->map_push_elem(&val_to_push, BPF_EXIST);

			// Brief sleep to increase chance of thread interleaving
			std::this_thread::sleep_for(std::chrono::microseconds(100)); // microseconds

			int popped_val;
			// Try to pop, ignore ENOENT as other threads might empty it.
			q_map_conc->map_pop_elem(&popped_val);
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

	// Concurrency test assertions are tricky.
	// A simple check could be that the map is not in a corrupted state,
	// or that the total number of elements makes sense if all ops were tracked.
	// For this conceptual test, we mainly rely on the internal mutex of queue_map_impl
	// to prevent crashes. More rigorous testing would involve specific invariants.
	// Example: try to empty the queue and check it becomes empty.
	int final_pop_val;
	int pop_count = 0;
	while(q_map_conc->map_pop_elem(&final_pop_val) == 0) {
		pop_count++;
	}
	INFO("Final pop count from concurrent queue: " << pop_count);
	REQUIRE(q_map_conc->map_pop_elem(&final_pop_val) == -ENOENT); // Must be empty eventually

	if (q_map_conc) {
		shm.destroy_ptr(q_map_conc);
	}
}
