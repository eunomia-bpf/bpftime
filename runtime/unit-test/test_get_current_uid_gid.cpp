/* SPDX-License-Identifier: MIT */
#include <catch2/catch_test_macros.hpp>
#include <boost/scope_exit.hpp>
#include <cstdint>
#include <unistd.h>

namespace
{
struct credentials {
	uid_t uid;
	gid_t gid;
};

thread_local const credentials *current_credentials = nullptr;
} // namespace

extern "C" {
uint64_t bpf_get_current_uid_gid(uint64_t, uint64_t, uint64_t, uint64_t,
				 uint64_t);
uid_t __real_getuid() noexcept;
gid_t __real_getgid() noexcept;

// Change query results without changing the test process's real credentials.
// Calls outside the scoped override, including other tests, use libc.
uid_t __wrap_getuid() noexcept
{
	return current_credentials ? current_credentials->uid : __real_getuid();
}

gid_t __wrap_getgid() noexcept
{
	return current_credentials ? current_credentials->gid : __real_getgid();
}
}

TEST_CASE("bpf_get_current_uid_gid reports current unsigned credentials",
	  "[uid_gid]")
{
	{
		const auto *previous = current_credentials;
		BOOST_SCOPE_EXIT_ALL(=)
		{
			current_credentials = previous;
		};
		const struct {
			credentials ids;
			uint64_t expected;
		} cases[] = {
			{ { 1001, 2002 }, 0x000007d2000003e9ULL },
			{ { 3003, 2002 }, 0x000007d200000bbbULL },
			{ { 3003, 4004 }, 0x00000fa400000bbbULL },
			{ { 0, 0 }, 0 },
			{ { 0x80000001U, 42 }, 0x0000002a80000001ULL },
			{ { 42, 0x80000002U }, 0x800000020000002aULL },
		};
		// Keep calls in one process to detect cached UID or GID values.
		for (const auto &test : cases) {
			current_credentials = &test.ids;
			CAPTURE(test.ids.uid, test.ids.gid);
			CHECK(bpf_get_current_uid_gid(0, 0, 0, 0, 0) ==
			      test.expected);
		}
	}
	const auto actual = bpf_get_current_uid_gid(0, 0, 0, 0, 0);
	CHECK(static_cast<uint32_t>(actual) == __real_getuid());
	CHECK(static_cast<uint32_t>(actual >> 32) == __real_getgid());
}
