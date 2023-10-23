#include <fcntl.h>
#include <unistd.h>
#include <frida-gum.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>
#include <inttypes.h>
#include <dlfcn.h>
#include "bpftime.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_handler.hpp"

using namespace bpftime;
using namespace boost::interprocess;

const shm_open_type bpftime::global_shm_open_type = shm_open_type::SHM_REMOVE_AND_CREATE;

int main(int argc, const char **argv)
{
	bpftime::bpf_map_attr attr;
	attr.type = bpf_map_handler::BPF_MAP_TYPE_HASH;
	bpftime_maps_create("test", attr);
	return system(
		(std::string("./test_shm_client_Tests") + " sub").c_str());
}
