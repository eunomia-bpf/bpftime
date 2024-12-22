#include "catch2/catch_test_macros.hpp"
#include "spdlog/spdlog.h"
#include <cstdlib>
#include <cstring>
#include <unistd.h>

extern "C" {

uint64_t bpftime_probe_read(uint64_t dst, uint64_t size, uint64_t ptr, uint64_t,
			    uint64_t);
uint64_t bpftime_probe_write_user(uint64_t dst, uint64_t src, uint64_t len,
				  uint64_t, uint64_t);

// prepare for future use
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

TEST_CASE("Test bpftime_probe_read") // test for bpftime_probe_read
{
	int dst[4] = { 0 };
	int src[4] = { 1, 2, 3, 4 };
	uint64_t size = sizeof(src);
	int64_t ret =
		bpftime_probe_read((uint64_t)dst, size, (uint64_t)src, 0, 0);
	REQUIRE(ret == 0);
	size_t len = sizeof(src) / sizeof(src[0]);
	for (size_t i = 0; i < len; i++) {
		REQUIRE(dst[i] == src[i]);
	}
	ret = bpftime_probe_read((uint64_t)dst, size, (uint64_t)(nullptr), 0, 0);
	REQUIRE(ret == -EFAULT);

	ret = 0;
	ret = bpftime_probe_read((uint64_t)(nullptr), size, (uint64_t)(nullptr), 0, 0);
	REQUIRE(ret == -EFAULT);
}

TEST_CASE("Test bpftime_probe_write_user") // test for bpftime_probe_write_user
{
	int dst[4] = { 0 };
	int src[4] = { 1, 2, 3, 4 };
	uint64_t size = sizeof(src);
	int64_t ret = bpftime_probe_write_user((uint64_t)dst, (uint64_t)src, size,
					   0, 0);
	REQUIRE(ret == 0);
	size_t len = 4;
	for (size_t i = 0; i < len; i++) {
		REQUIRE(dst[i] == src[i]);
	}

	ret = bpftime_probe_write_user((uint64_t)(nullptr), (uint64_t)(src), size,
				       0, 0);
	REQUIRE(ret == -EFAULT);

	ret = bpftime_probe_write_user((uint64_t)dst, (uint64_t)(nullptr), size, 0,
				       0);
	REQUIRE(ret == -EFAULT);

	void *dst1 = (void *)(dst);
	void *src1 = (void *)(src);
	ret = bpftime_probe_write_user((uint64_t)dst1, (uint64_t)src1, size, 0,
				       0);
	REQUIRE(ret == 0);
	for (size_t i = 0; i < len; i++) {
		REQUIRE(((int *)dst1)[i] == ((int *)src1)[i]);
	}
}