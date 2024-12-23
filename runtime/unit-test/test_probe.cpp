#include "catch2/catch_test_macros.hpp"
#include "spdlog/spdlog.h"
#include <csetjmp>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <signal.h>

extern "C" {

uint64_t bpftime_probe_read(uint64_t dst, uint64_t size, uint64_t ptr, uint64_t,
			    uint64_t);
uint64_t bpftime_probe_write_user(uint64_t dst, uint64_t src, uint64_t len,
				  uint64_t, uint64_t);
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
	ret = bpftime_probe_read((uint64_t)dst, size, (uint64_t)(nullptr), 0,
				 0);
	REQUIRE(ret == -EFAULT);

	ret = 0;
	ret = bpftime_probe_read((uint64_t)(nullptr), size, (uint64_t)(nullptr),
				 0, 0);
	REQUIRE(ret == -EFAULT);
}

TEST_CASE("Test bpftime_probe_write_user") // test for bpftime_probe_write_user
{
	int dst[4] = { 0 };
	int src[4] = { 1, 2, 3, 4 };
	uint64_t size = sizeof(src);
	int64_t ret = bpftime_probe_write_user((uint64_t)dst, (uint64_t)src,
					       size, 0, 0);
	REQUIRE(ret == 0);
	size_t len = 4;
	for (size_t i = 0; i < len; i++) {
		REQUIRE(dst[i] == src[i]);
	}

	ret = bpftime_probe_write_user((uint64_t)(nullptr), (uint64_t)(src),
				       size, 0, 0);
	REQUIRE(ret == -EFAULT);

	ret = bpftime_probe_write_user((uint64_t)dst, (uint64_t)(nullptr), size,
				       0, 0);
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

TEST_CASE("Test Probe read/write size valid or not ")
{
	int dst[4] = { 0 };
	int src[4] = { 1, 2, 3, 4 };
	uint64_t size = sizeof(src);
	int64_t ret =
		bpftime_probe_read((uint64_t)dst, -1, (uint64_t)src, 0, 0);
	REQUIRE(ret == -EFAULT);
	ret = bpftime_probe_write_user((uint64_t)dst, (uint64_t)src, -1, 0, 0);
	REQUIRE(ret == -EFAULT);
	ret = bpftime_probe_read((uint64_t)dst, -1, (uint64_t)(nullptr), 0, 0);
	REQUIRE(ret == -EFAULT);
	ret = bpftime_probe_read((uint64_t)dst, size, (uint64_t)(src), 0, 0);
	REQUIRE(ret == 0);
	for (size_t i = 0; i < 4; i++) {
		REQUIRE(dst[i] == src[i]);
	}
}


TEST_CASE("Test origin handler is null")
{
	struct sigaction original_sa, sa;
	int dst[4] = { 0 };
	int src[4] = { 1, 2, 3, 4 };
	sa.sa_flags = SA_SIGINFO;
	sigemptyset(&sa.sa_mask);
	sa.sa_sigaction = nullptr;
	if (sigaction(SIGSEGV, &sa, nullptr) < 0) {
		REQUIRE(false);
	}
	uint64_t size = sizeof(src);

	int ret = bpftime_probe_read((uint64_t)(dst), size, (uint64_t)(src), 0,
				     0);
	REQUIRE(ret == 0);

	sigaction(SIGSEGV, nullptr, &original_sa);
	auto handler = original_sa.sa_sigaction;
	REQUIRE(handler != nullptr);
}