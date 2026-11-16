/* SPDX-License-Identifier: MIT */
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <error.h>
#include <vector>

extern "C" {
struct xdp_md_userspace {
	uint64_t data;
	uint64_t data_end;
	uint32_t data_meta;
	uint32_t ingress_ifindex;
	uint32_t rx_queue_index;
	uint32_t egress_ifindex;
	uint64_t buffer_start;
	uint64_t buffer_end;
};

long bpftime_xdp_adjust_head(struct xdp_md_userspace *xdp, int offset);
}

/* A packet of `len` bytes whose payload is a recognisable ramp, sitting at
 * `head` inside a buffer that has `headroom` bytes of slack in front of it. */
struct fixture {
	std::vector<uint8_t> buf;
	xdp_md_userspace xdp;

	explicit fixture(size_t headroom, size_t len) : buf(headroom + len + 64)
	{
		for (size_t i = 0; i < buf.size(); i++)
			buf[i] = (uint8_t)(i % 251);
		std::memset(&xdp, 0, sizeof(xdp));
		xdp.buffer_start = (uint64_t)(uintptr_t)buf.data();
		xdp.buffer_end = xdp.buffer_start + buf.size();
		xdp.data = xdp.buffer_start + headroom;
		xdp.data_end = xdp.data + len;
	}

	const uint8_t *payload() const { return (const uint8_t *)(uintptr_t)xdp.data; }
	size_t len() const { return (size_t)(xdp.data_end - xdp.data); }
};

/* Growing the head inside the existing headroom moves no bytes: the head
 * moves back by |offset| and the payload stays where it was, so the length
 * reported by data..data_end grows by |offset|. */
TEST_CASE("xdp_adjust_head: negative offset inside headroom keeps the bytes", "[xdp]")
{
	fixture f(64, 128);
	const uint8_t *payload_before = f.payload();
	std::vector<uint8_t> original(payload_before, payload_before + f.len());

	REQUIRE(bpftime_xdp_adjust_head(&f.xdp, -16) == 0);
	REQUIRE(f.len() == 128 + 16);
	/* The 128 original bytes are still at the same address, now 16 into the
	 * new data pointer. */
	REQUIRE(f.payload() + 16 == payload_before);
	REQUIRE(std::memcmp(f.payload() + 16, original.data(), original.size()) == 0);
}

/* Expanding past buffer_start must relocate the packet to the requested head,
 * not by the distance to buffer_start: those differ when offset is smaller
 * than the gap, and the earlier form left data pointing at bytes the packet
 * had not been moved to. */
TEST_CASE("xdp_adjust_head: negative offset before buffer_start relocates to the head",
	  "[xdp]")
{
	fixture f(8, 96);
	std::vector<uint8_t> original(f.payload(), f.payload() + f.len());

	/* Ask for 32 bytes of head; only 8 exist, so the head stops at
	 * buffer_start, the packet moves up by 8, and the length grows by 8. */
	REQUIRE(bpftime_xdp_adjust_head(&f.xdp, -32) == 0);
	REQUIRE(f.xdp.data == f.xdp.buffer_start);
	REQUIRE(f.len() == 96 + 8);
	REQUIRE(std::memcmp(f.payload(), original.data(), original.size()) == 0);
}

/* If the packet cannot fit at buffer_start, the call has to fail rather than
 * write out of bounds. The fixture is deliberately degenerate: the packet is
 * longer than the whole buffer, which the old form did not check at all. */
TEST_CASE("xdp_adjust_head: relocation that does not fit is rejected", "[xdp]")
{
	std::vector<uint8_t> buf(64);
	xdp_md_userspace xdp{};
	xdp.buffer_start = (uint64_t)(uintptr_t)buf.data();
	xdp.buffer_end = xdp.buffer_start + buf.size();
	xdp.data = xdp.buffer_start + 8;
	xdp.data_end = xdp.data + 60; /* 60-byte packet ... */
	xdp.buffer_end = xdp.buffer_start + 32; /* ... in a 32-byte buffer */

	/* An offset of -16 asks for a head before buffer_start, so the packet
	 * must be moved there; relocating 60 bytes would run past buffer_end. */
	REQUIRE(bpftime_xdp_adjust_head(&xdp, -16) == -EINVAL);
}
