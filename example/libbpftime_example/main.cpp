#include "bpftime_shm.hpp"
#include <iostream>

int main() {
    // Initialize global shared memory with SHM_CREATE_OR_OPEN option
    bpftime_initialize_global_shm(bpftime::shm_open_type::SHM_CREATE_OR_OPEN);

    // Create a shared memory segment with a name and size
    const char* shmName = "example_shm";
    const size_t shmSize = 1024; // 1KB
    boost::interprocess::managed_shared_memory shm(boost::interprocess::create_only, shmName, shmSize);

    // Allocate a string in the shared memory
    typedef boost::interprocess::allocator<char, boost::interprocess::managed_shared_memory::segment_manager> CharAllocator;
    typedef boost::interprocess::basic_string<char, std::char_traits<char>, CharAllocator> SharedString;
    SharedString* sharedStr = shm.construct<SharedString>("shared_string")(shm.get_segment_manager());

    // Write a string to the shared memory
    *sharedStr = "Hello from shared memory!";

    // Read the string back from shared memory and print it
    std::cout << "String read from shared memory: " << *sharedStr << std::endl;

    bpftime_destroy_global_shm();
    // Remove the shared memory segment
    bpftime_remove_global_shm();

    return 0;
}
