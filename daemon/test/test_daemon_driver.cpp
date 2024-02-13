/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include "user/bpftime_driver.hpp"
#if !defined(__x86_64__) && defined(_M_X64)
#error Only supports x86_64
#endif

#define TEST_INSN_SIZE 12

TEST_CASE("Test daemon driver")
{
	daemon_config cfg;
	char insns[12 * sizeof(ebpf_inst)];
	int pid_0 = 0;
	int pid_1 = 1;
    int id;

	// bpftime_driver driver(cfg);
	// id = driver.bpftime_progs_create_server(pid_0, 1);
    // REQUIRE(id >= 0);
	// id = driver.bpftime_progs_create_server(pid_1, 1);
    // REQUIRE(id >= 0);
}
