/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, bpftime contributors
 * All rights reserved.
 */
#include <iostream>
#include <cassert>
#include <cstring>
#include <bpf_map/userspace/stack.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

using namespace bpftime;
using namespace boost::interprocess;

void test_stack_basic_operations()
{
	std::cout << "Testing basic stack operations..." << std::endl;

	// Create shared memory
	managed_shared_memory memory(create_only, "test_stack_shm",
				     1024 * 1024);

	// Create stack map
	stack_map_impl stack(memory, sizeof(int), 5);

	// Test empty stack
	assert(stack.get_current_size() == 0);
	assert(stack.elem_lookup(nullptr) == nullptr);
	assert(stack.map_peek_elem(nullptr) == -EINVAL);

	int value;
	assert(stack.map_pop_elem(&value) == -ENOENT);

	// Test push operations
	int values[] = { 10, 20, 30, 40, 50 };

	for (int i = 0; i < 5; i++) {
		assert(stack.map_push_elem(&values[i], 0) == 0);
		assert(stack.get_current_size() == i + 1);
	}

	// Test stack is full
	int extra = 60;
	assert(stack.map_push_elem(&extra, 0) == -E2BIG);

	// Test peek (should return top element: 50)
	assert(stack.map_peek_elem(&value) == 0);
	assert(value == 50);
	assert(stack.get_current_size() == 5); // Size unchanged after peek

	// Test pop operations (LIFO order: 50, 40, 30, 20, 10)
	for (int i = 4; i >= 0; i--) {
		assert(stack.map_pop_elem(&value) == 0);
		assert(value == values[i]);
		assert(stack.get_current_size() == i);
	}

	// Test empty stack again
	assert(stack.map_pop_elem(&value) == -ENOENT);

	std::cout << "Basic stack operations test passed!" << std::endl;

	// Cleanup
	shared_memory_object::remove("test_stack_shm");
}

void test_stack_with_flags()
{
	std::cout << "Testing stack operations with flags..." << std::endl;

	// Create shared memory
	managed_shared_memory memory(create_only, "test_stack_flags_shm",
				     1024 * 1024);

	// Create stack map
	stack_map_impl stack(memory, sizeof(int), 3);

	// Fill the stack
	int values[] = { 10, 20, 30 };
	for (int i = 0; i < 3; i++) {
		assert(stack.map_push_elem(&values[i], 0) == 0);
	}

	// Test BPF_EXIST flag when full (should remove oldest and add new)
	int new_value = 40;
	assert(stack.map_push_elem(&new_value, BPF_EXIST) == 0);
	assert(stack.get_current_size() == 3);

	// Pop and verify order: should be 40, 30, 20 (10 was removed)
	int value;
	assert(stack.map_pop_elem(&value) == 0);
	assert(value == 40);

	assert(stack.map_pop_elem(&value) == 0);
	assert(value == 30);

	assert(stack.map_pop_elem(&value) == 0);
	assert(value == 20);

	assert(stack.get_current_size() == 0);

	std::cout << "Stack operations with flags test passed!" << std::endl;

	// Cleanup
	shared_memory_object::remove("test_stack_flags_shm");
}

void test_stack_elem_operations()
{
	std::cout << "Testing stack elem_* operations..." << std::endl;

	// Create shared memory
	managed_shared_memory memory(create_only, "test_stack_elem_shm",
				     1024 * 1024);

	// Create stack map
	stack_map_impl stack(memory, sizeof(int), 3);

	// Test elem_update (push)
	int values[] = { 100, 200, 300 };
	for (int i = 0; i < 3; i++) {
		assert(stack.elem_update(nullptr, &values[i], BPF_ANY) == 0);
	}

	// Test elem_lookup (peek)
	void *ptr = stack.elem_lookup(nullptr);
	assert(ptr != nullptr);
	int *top_value = static_cast<int *>(ptr);
	assert(*top_value == 300); // Top of stack

	// Test elem_delete (pop)
	assert(stack.elem_delete(nullptr) == 0);
	assert(stack.get_current_size() == 2);

	// Verify new top
	ptr = stack.elem_lookup(nullptr);
	assert(ptr != nullptr);
	top_value = static_cast<int *>(ptr);
	assert(*top_value == 200);

	std::cout << "Stack elem_* operations test passed!" << std::endl;

	// Cleanup
	shared_memory_object::remove("test_stack_elem_shm");
}

int main()
{
	try {
		test_stack_basic_operations();
		test_stack_with_flags();
		test_stack_elem_operations();

		std::cout << "All stack map tests passed!" << std::endl;
		return 0;
	} catch (const std::exception &e) {
		std::cerr << "Test failed with exception: " << e.what()
			  << std::endl;
		return 1;
	}
}