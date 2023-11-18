/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <attach/attach_manager/frida_attach_manager.hpp>
#if !defined(__x86_64__) && defined(_M_X64)
#error Only supports x86_64
#endif
using namespace bpftime;


TEST_CASE("Test daemon")
{
	
}
