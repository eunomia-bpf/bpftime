/* SPDX-License-Identifier: MIT */
#include <catch2/catch_test_macros.hpp>

#include "../common_def.hpp"
#include <bpf_map/userspace/ringbuf_map.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <vector>

namespace
{
constexpr size_t HDR_SZ = 8;
constexpr int MAP_FD = 42;

int collect(void *ctx, void *data, size_t len)
{
	auto *out = static_cast<std::vector<std::vector<uint8_t>> *>(ctx);
	auto *p = static_cast<uint8_t *>(data);
	out->emplace_back(p, p + len);
	return 0;
}

void produce(bpftime::ringbuf &rb, size_t size, uint8_t fill)
{
	void *sample = rb.reserve(size, MAP_FD);
	REQUIRE(sample != nullptr);
	// bpf_ringbuf_submit() recovers the map fd from the word right before
	// the sample, so it must be the fd stored in the header
	REQUIRE(static_cast<int32_t *>(sample)[-1] == MAP_FD);
	std::memset(sample, fill, size);
	rb.submit(sample, false);
}
} // namespace

TEST_CASE("Ringbuf record whose header is the last 8 bytes of the ring is "
	  "delivered",
	  "[ringbuf_map]")
{
	const char *SHM_NAME = "RingbufMapWrapTestShmCatch2";
	shm_remove remover((std::string(SHM_NAME)));
	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHM_NAME, 1 << 20);

	const uint32_t max_ent = getpagesize();
	bpftime::ringbuf_map_impl map(max_ent, shm);
	auto rb = map.create_impl_shared_ptr();

	// 48 byte payloads take 56 bytes on the ring, and 56 divides
	// max_ent - 8 for 4 KiB pages: after filling the ring up to that point
	// the next header sits in the last 8 bytes.
	const size_t record = 48 + HDR_SZ;
	REQUIRE((max_ent - HDR_SZ) % record == 0);
	std::vector<std::vector<uint8_t>> got;
	for (size_t off = 0; off < max_ent - HDR_SZ; off += record)
		produce(*rb, 48, 0x11);
	REQUIRE(rb->fetch_data(collect, &got) ==
		(int)((max_ent - HDR_SZ) / record));

	// This record straddles the end of the ring
	got.clear();
	produce(*rb, 16, 0xAB);
	produce(*rb, 16, 0xCD);
	REQUIRE(rb->fetch_data(collect, &got) == 2);
	REQUIRE(got[0] == std::vector<uint8_t>(16, 0xAB));
	REQUIRE(got[1] == std::vector<uint8_t>(16, 0xCD));
}
