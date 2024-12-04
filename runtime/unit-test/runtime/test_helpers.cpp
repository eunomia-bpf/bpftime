#include "bpftime_prog.hpp"
#include "catch2/catch_test_macros.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <optional>
#include <unistd.h>
extern "C" {
uint64_t bpftime_probe_read(uint64_t dst, uint64_t size, uint64_t ptr, uint64_t,
			    uint64_t);
uint64_t bpftime_probe_write_user(uint64_t dst, uint64_t src, uint64_t len,
				  uint64_t, uint64_t);
long bpftime_strncmp(const char *s1, uint64_t s1_sz, const char *s2);
uint64_t bpftime_get_prandom_u32(void);
uint64_t bpftime_ktime_get_coarse_ns(uint64_t, uint64_t, uint64_t, uint64_t,
				     uint64_t);
uint64_t bpf_ktime_get_coarse_ns(uint64_t, uint64_t, uint64_t, uint64_t,
				 uint64_t);
uint64_t bpftime_ktime_get_ns(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
uint64_t bpftime_get_current_pid_tgid(uint64_t, uint64_t, uint64_t, uint64_t,
				      uint64_t);
uint64_t bpf_get_current_uid_gid(uint64_t, uint64_t, uint64_t, uint64_t,
				 uint64_t);
uint64_t bpftime_get_current_comm(uint64_t buf, uint64_t size, uint64_t,
				  uint64_t, uint64_t);
uint64_t bpf_probe_read_str(uint64_t buf, uint64_t bufsz, uint64_t ptr,
			    uint64_t, uint64_t);
uint64_t bpftime_get_smp_processor_id();
uint64_t bpftime_get_attach_cookie(uint64_t ctx, uint64_t, uint64_t, uint64_t,
				   uint64_t);

uint64_t bpftime_get_smp_processor_id();
}
TEST_CASE("Test helpers strncmp", "[helper]")
{
	char s1[] = "aaa";
	char s2[] = "aaa";
	char s3[] = "aab";
	REQUIRE(bpftime_strncmp(s1, 4, s2) == 0);
	REQUIRE(bpftime_strncmp(s1, 4, s3) != 0);
}

TEST_CASE("Test helpers probe_read/probe_write_user/probe_read_str", "[helper]")
{
	char buf1[] = "aaaaa";
	char buf2[sizeof(buf1)];

	SECTION("probe read")
	{
		REQUIRE(bpftime_probe_read((uintptr_t)&buf2, sizeof(buf1),
					   (uintptr_t)&buf1, 0, 0) == 0);
	}
	SECTION("probe write user")
	{
		REQUIRE(bpftime_probe_write_user((uintptr_t)&buf2,
						 (uintptr_t)&buf1, sizeof(buf1),
						 0, 0) == 0);
	}
	SECTION("bpf_probe_read_str")
	{
		REQUIRE(bpf_probe_read_str((uintptr_t)&buf2, sizeof(buf1),
					   (uintptr_t)&buf1, 0, 0) == 0);
	}
	REQUIRE(memcmp(buf1, buf2, sizeof(buf1)) == 0);
}
TEST_CASE("Test helpers get_prandom_u32", "[helper]")
{
	bpftime_get_prandom_u32();
}
TEST_CASE("Test helpers time functions", "[helper]")
{
	auto val1 = bpftime_ktime_get_coarse_ns(0, 0, 0, 0, 0);
	auto val2 = bpf_ktime_get_coarse_ns(0, 0, 0, 0, 0);
	auto val3 = bpftime_ktime_get_ns(0, 0, 0, 0, 0);
	uint64_t vals[] = { val1, val2, val3 };
	std::sort(std::begin(vals), std::end(vals));
	REQUIRE(vals[1] - vals[0] < 1e9);
	REQUIRE(vals[2] - vals[1] < 1e9);
}

TEST_CASE("Test helpers get pid tgid", "[helper]")
{
	auto result = bpftime_get_current_pid_tgid(0, 0, 0, 0, 0);
	auto tid = (int32_t)result;
	int32_t tgid = result >> 32;
	REQUIRE(tid == gettid());
	REQUIRE(tgid == getpid());
}

TEST_CASE("Test helpers get_uid_gid", "[helper]")
{
	auto result = bpf_get_current_uid_gid(0, 0, 0, 0, 0);
	uint32_t uid = result;
	uint32_t gid = result >> 32;
	REQUIRE(uid == getuid());
	REQUIRE(gid == getgid());
}

TEST_CASE("Test helpers get_current_comm", "[helper]")
{
	char buf[100];
	REQUIRE(bpftime_get_current_comm((uintptr_t)&buf, sizeof(buf), 0, 0,
					 0) == 0);
	REQUIRE(strcmp(buf, "bpftime_runtime_tests") == 0);
}

TEST_CASE("Test helpers get_attach_cookie", "[helper]")
{
	bpftime::current_thread_bpf_cookie = 0x12345678;
	REQUIRE(bpftime_get_attach_cookie(0, 0, 0, 0, 0) == 0x12345678);
	bpftime::current_thread_bpf_cookie.reset();
	REQUIRE(bpftime_get_attach_cookie(0, 0, 0, 0, 0) == 0);
}

TEST_CASE("Test helpers get_smp_processor_id", "[helper]")
{
	cpu_set_t orig, set;
	CPU_ZERO(&orig);
	CPU_ZERO(&set);
	REQUIRE(sched_getaffinity(0, sizeof(orig), &orig) == 0);
	CPU_SET(0, &set);

	REQUIRE(sched_setaffinity(0, sizeof(set), &set) == 0);
	REQUIRE(bpftime_get_smp_processor_id() == 0);
	REQUIRE(sched_setaffinity(0, sizeof(orig), &orig) == 0);
}
