/* SPDX-License-Identifier: MIT */
#include <catch2/catch_test_macros.hpp>

#if __linux__
#include "bpftime_shm.hpp"
#include "common_def.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <boost/scope_exit.hpp>
#include <linux/bpf.h>
#include <string>
#include <unistd.h>
#include <vector>

// Ringbuf records are 8-byte headers followed by the payload, both kept
// adjacent in a power-of-two data area. When a header lands in the very
// last 8 bytes of the ring, the payload no longer fits behind it inside
// the ring; the record must be kept contiguous instead of wrapping the
// payload back to the start of the data area.
TEST_CASE("ringbuf keeps tail-straddling records contiguous and consumable",
	  "[ringbuf]")
{
	const char *old_name = getenv("BPFTIME_GLOBAL_SHM_NAME");
	const bool had_name = old_name != nullptr;
	const std::string previous_name = old_name ? old_name : "";
	const std::string name = "RingbufTailShm-" + std::to_string(getpid());
	shm_remove remover{ std::string(name) };
	bool initialized = false;
	BOOST_SCOPE_EXIT_ALL(&)
	{
		if (initialized) {
			bpftime_destroy_global_shm();
			bpftime_initialize_global_shm(
				bpftime::shm_open_type::SHM_NO_CREATE);
		}
		if (had_name)
			setenv("BPFTIME_GLOBAL_SHM_NAME", previous_name.c_str(),
			       1);
		else
			unsetenv("BPFTIME_GLOBAL_SHM_NAME");
	};
	REQUIRE(setenv("BPFTIME_GLOBAL_SHM_NAME", name.c_str(), 1) == 0);
	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
	initialized = true;

	struct rec {
		uint64_t seq;
		uint8_t pad[40];
	};
	static_assert(sizeof(rec) == 48, "reproducer record size");

	// 48-byte records occupy 56 bytes on the ring (48 payload + 8
	// header), and 56 * 73 = 4088 = 4096 - 8, so record 74 puts its
	// header in the last 8 bytes of the ring.
	const int rb_fd = bpftime_maps_create(
		-1, "tail_wrap_ring",
		{ .type = BPF_MAP_TYPE_RINGBUF, .max_ents = 4096 });
	REQUIRE(rb_fd >= 0);

	uint64_t received = 0;
	auto drain = +[](void *ctx, void *data, size_t len) -> int {
		REQUIRE(len == sizeof(rec));
		auto *out = static_cast<uint64_t *>(ctx);
		const auto *r = static_cast<const rec *>(data);
		REQUIRE(r->seq == *out);
		(*out)++;
		return 0;
	};

	// Reserve + submit one record at a time and drain it, so every
	// record is written exactly at the current producer offset and the
	// wrap position is exercised deterministically.
	for (uint64_t seq = 0; seq < 200; ++seq) {
		auto *payload = bpftime_ringbuf_reserve(rb_fd, sizeof(rec));
		REQUIRE(payload != nullptr);
		auto *r = static_cast<rec *>(payload);
		r->seq = seq;
		for (auto &b : r->pad)
			b = 0;
		bpftime_ringbuf_submit(rb_fd, payload, 0);

		const int drained =
			bpftime_poll_from_ringbuf(rb_fd, &received, drain);
		REQUIRE(drained == 1);
		REQUIRE(received == seq + 1);
	}
}

#endif /* __linux__ */
