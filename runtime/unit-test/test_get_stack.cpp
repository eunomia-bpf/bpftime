#include "linux/bpf.h"

#include <catch2/catch_test_macros.hpp>

#include <cerrno>
#include <cstdint>

extern "C" int64_t bpftime_get_stack(uint64_t, uint64_t, uint64_t, uint64_t,
				     uint64_t);

TEST_CASE("get_stack rejects build IDs")
{
	REQUIRE(bpftime_get_stack(0, 0, 0,
				  BPF_F_USER_STACK | BPF_F_USER_BUILD_ID,
				  0) == -ENOTSUP);
}
