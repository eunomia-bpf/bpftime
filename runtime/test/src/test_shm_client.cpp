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
	if (argc == 1) {
		return 0;
	}
	int res = 1;
	bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);

	bpf_attach_ctx ctx;
	agent_config config;
	config.enable_ufunc_helper_group = true;
	res = ctx.init_attach_ctx_from_handlers(config);
	if (res != 0) {
		return res;
	}
	// don't free ctx here
	return 0;
}
