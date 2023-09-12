#include <cstdlib>
#include "bpftime_handler.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <string>
#include <linux/bpf.h>

const char *SHM_NAME = "my_shm_maps_test";
const char *HANDLER_NAME = "my_handler";

using namespace boost::interprocess;
using namespace bpftime;

const shm_open_type bpftime::global_shm_open_type = shm_open_type::SHM_NO_CREATE;

void test_insert_map(int fd, bpftime::handler_manager &manager_ref,
		     managed_shared_memory &segment)
{
	auto &map_handler = std::get<bpf_map_handler>(manager_ref[fd]);
	for (int i = 0; i < 100; i++) {
		uint32_t key = i;
		uint64_t value = (((uint64_t)key) << 32) | 0xffffffff;
		map_handler.map_update_elem(&key, &value, 0);
	}
}

void test_lookup_map(int fd, bpftime::handler_manager &manager_ref,
		     managed_shared_memory &segment)
{
	auto &map_handler = std::get<bpf_map_handler>(manager_ref[fd]);
	for (int i = 0; i < 100; i++) {
		uint32_t key = i;
		auto val = *(uint64_t *)(map_handler.map_lookup_elem(&key));
		assert(val == ((((uint64_t)key) << 32) | 0xffffffff));
		std::cout << "val for " << i << " = " << std::hex << val
			  << std::endl;
	}
}

void test_delete_map(int fd, bpftime::handler_manager &manager_ref,
		     managed_shared_memory &segment)
{
	auto &map_handler = std::get<bpf_map_handler>(manager_ref[fd]);
	for (int i = 0; i < 100; i++) {
		uint64_t key = i;
		map_handler.map_delete_elem(&key);
	}
}

void test_get_next_element(int fd, bpftime::handler_manager &manager_ref,
			   managed_shared_memory &segment)
{
	auto &map_handler = std::get<bpf_map_handler>(manager_ref[fd]);
	uint32_t key = 0;
	uint32_t next_key = 0;
	while (map_handler.bpf_map_get_next_key(&key, &next_key) == 0) {
		std::cout << "key = " << key << ", next_key = " << next_key
			  << std::endl;
		key = next_key;
	}
	std::cout << "key = " << key << ", next_key = " << next_key
		  << std::endl;
}

int main(int argc, const char **argv)
{
	if (argc == 1) {
		struct shm_remove remover(SHM_NAME);

		// The side that creates the mapping
		managed_shared_memory segment(create_only, SHM_NAME, 1 << 20);
		auto manager = segment.construct<handler_manager>(HANDLER_NAME)(
			segment);
		auto &manager_ref = *manager;

		manager_ref.set_handler(1,
					bpf_map_handler(BPF_MAP_TYPE_HASH, 4, 8,
							1024, 0, "hash1",
							segment),
					segment);
		manager_ref.set_handler(2,
					bpf_map_handler(BPF_MAP_TYPE_HASH, 4, 8,
							1024, 0, "hash2",
							segment),
					segment);
		manager_ref.set_handler(3,
					bpf_map_handler(BPF_MAP_TYPE_ARRAY, 4,
							8, 1024, 0, "array1",
							segment),
					segment);

		// test insert
		test_insert_map(1, manager_ref, segment);
		test_insert_map(3, manager_ref, segment);
		test_get_next_element(1, manager_ref, segment);
		test_get_next_element(3, manager_ref, segment);
		// test lookup
		test_lookup_map(1, manager_ref, segment);
		test_lookup_map(3, manager_ref, segment);
		std::cout << "Starting sub process" << std::endl;
		system((std::string(argv[0]) + " sub").c_str());
		assert(segment.find<handler_manager>(HANDLER_NAME).first ==
		       nullptr);
	} else {
		std::cout << "Subprocess started" << std::endl;
		managed_shared_memory segment(open_only, SHM_NAME);
		auto manager =
			segment.find<handler_manager>(HANDLER_NAME).first;
		auto &manager_ref = *manager;
		manager_ref.clear_fd_at(2, segment);
		test_lookup_map(1, manager_ref, segment);
		test_lookup_map(3, manager_ref, segment);
		test_get_next_element(1, manager_ref, segment);
		test_get_next_element(3, manager_ref, segment);
		test_delete_map(1, manager_ref, segment);
		test_delete_map(3, manager_ref, segment);
		manager->clear_all(segment);
		std::cout << "Print maps value finished" << std::endl;
		segment.destroy_ptr(manager);
		std::cout << "Subprocess exited" << std::endl;
	}
	return 0;
}
