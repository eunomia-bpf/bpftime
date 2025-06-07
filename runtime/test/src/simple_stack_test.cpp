#include <iostream>
#include <cassert>
#include <bpf_map/userspace/stack.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

using namespace bpftime;
using namespace boost::interprocess;

int main()
{
	try {
		std::cout << "Creating shared memory..." << std::endl;

		// Remove any existing shared memory
		shared_memory_object::remove("simple_stack_test");

		// Create shared memory
		managed_shared_memory memory(create_only, "simple_stack_test",
					     1024 * 1024);

		std::cout << "Creating stack map..." << std::endl;

		// Create stack map with capacity 3
		stack_map_impl stack(memory, sizeof(int), 3);

		std::cout << "Testing basic operations..." << std::endl;

		// Test empty stack
		assert(stack.get_current_size() == 0);
		std::cout << "✓ Empty stack size correct" << std::endl;

		// Push some values
		int val1 = 10, val2 = 20, val3 = 30;

		assert(stack.map_push_elem(&val1, 0) == 0);
		assert(stack.get_current_size() == 1);
		std::cout << "✓ Push 10 successful" << std::endl;

		assert(stack.map_push_elem(&val2, 0) == 0);
		assert(stack.get_current_size() == 2);
		std::cout << "✓ Push 20 successful" << std::endl;

		assert(stack.map_push_elem(&val3, 0) == 0);
		assert(stack.get_current_size() == 3);
		std::cout << "✓ Push 30 successful" << std::endl;

		// Test peek (should return 30 - top of stack)
		int peek_value;
		assert(stack.map_peek_elem(&peek_value) == 0);
		assert(peek_value == 30);
		assert(stack.get_current_size() == 3); // Size unchanged
		std::cout << "✓ Peek returned 30 (top of stack)" << std::endl;

		// Test pop (LIFO order: 30, 20, 10)
		int pop_value;
		assert(stack.map_pop_elem(&pop_value) == 0);
		assert(pop_value == 30);
		assert(stack.get_current_size() == 2);
		std::cout << "✓ Pop returned 30" << std::endl;

		assert(stack.map_pop_elem(&pop_value) == 0);
		assert(pop_value == 20);
		assert(stack.get_current_size() == 1);
		std::cout << "✓ Pop returned 20" << std::endl;

		assert(stack.map_pop_elem(&pop_value) == 0);
		assert(pop_value == 10);
		assert(stack.get_current_size() == 0);
		std::cout << "✓ Pop returned 10" << std::endl;

		std::cout
			<< "All tests passed! Stack map implementation is working correctly."
			<< std::endl;

		// Cleanup
		shared_memory_object::remove("simple_stack_test");

		return 0;
	} catch (const std::exception &e) {
		std::cerr << "Test failed with exception: " << e.what()
			  << std::endl;
		shared_memory_object::remove("simple_stack_test");
		return 1;
	}
}