/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
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

int main(int argc, const char **argv)
{
	bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);
	bpftime::bpf_map_attr attr;
	attr.type = bpf_map_type::BPF_MAP_TYPE_HASH;
	bpftime_maps_create(-1, "test", attr);
	return system(
		(std::string("./test_shm_client_Tests") + " sub").c_str());
}
