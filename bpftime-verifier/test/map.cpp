#include "bpf/bpf.h"
#include "catch2/catch_message.hpp"
#include "catch2/matchers/catch_matchers.hpp"
#include "linux/bpf.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <bpftime-verifier.hpp>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <ostream>
#include <catch2/matchers/catch_matchers_string.hpp>
using namespace bpftime;
using namespace verifier;
using Catch::Matchers::ContainsSubstring;
/*

A program reads from map but without null check

0000000000000000 <handle_exec>:
       0:	b7 01 00 00 02 00 00 00	r1 = 2
       1:	7b 1a f8 ff 00 00 00 00	*(u64 *)(r10 - 8) = r1
       2:	bf a2 00 00 00 00 00 00	r2 = r10
       3:	07 02 00 00 f8 ff ff ff	r2 += -8
       4:	18 11 00 00 1d 09 00 00
	    00 00 00 00 00 00 00 00	r1 = map_by_fd(2333)
       6:	85 00 00 00 01 00 00 00	call 1
       7:	61 00 00 00 00 00 00 00	r0 = *(u32 *)(r0 + 0)
       8:	95 00 00 00 00 00 00 00	exit
*/
const uint64_t prog_with_map_1[] = { 0x00000002000001b7, 0x00000000fff81a7b,
				     0x000000000000a2bf, 0xfffffff800000207,
				     0x0000091d00001118, 0x0000000000000000,
				     0x0000000100000085, 0x0000000000000061,
				     0x0000000000000095 };

TEST_CASE(
	"Test verifying ebpf programs with maps (with helpers and maps) with null check",
	"[maps][null-check]")
{
	// bpf_map_lookup_elem -> 1
	set_available_helpers(std::vector<int32_t>{ 1 });
	set_map_descriptors(std::map<int, BpftimeMapDescriptor>{
		{ 2333, BpftimeMapDescriptor{ .original_fd = 2333,
					      .type = BPF_MAP_TYPE_HASH,
					      .key_size = 8,
					      .value_size = 4,
					      .max_entries = 8192,
					      .inner_map_fd = 0 } } });
	auto ret =
		verify_ebpf_program(prog_with_map_1, std::size(prog_with_map_1),
				    "uprobe//proc/self/exe:uprobed_sub");
	REQUIRE(ret.has_value());
	REQUIRE_THAT(
		ret.value(),
		ContainsSubstring(
			"7: Possible null access (valid_access(r0.offset, width=4) for read)"));
}

/*
A program reads from map with null check
       0:	b7 01 00 00 02 00 00 00	r1 = 2
       1:	7b 1a f8 ff 00 00 00 00	*(u64 *)(r10 - 8) = r1
       2:	bf a2 00 00 00 00 00 00	r2 = r10
       3:	07 02 00 00 f8 ff ff ff	r2 += -8
       4:	18 11 00 00 1d 09 00 00
		00 00 00 00 00 00 00 00	r1 = map_by_fd(2333)
       6:	85 00 00 00 01 00 00 00	call 1
       7:	bf 01 00 00 00 00 00 00	r1 = r0
       8:	b7 00 00 00 01 00 00 00	r0 = 1
       9:	15 01 01 00 00 00 00 00	if r1 == 0 goto +1 <LBB0_2>
      10:	61 10 00 00 00 00 00 00	r0 = *(u32 *)(r1 + 0)
      11:	95 00 00 00 00 00 00 00	exit
*/
const uint64_t prog_with_map_2[] = { 0x00000002000001b7, 0x00000000fff81a7b,
				     0x000000000000a2bf, 0xfffffff800000207,
				     0x0000091d00001118, 0x0000000000000000,
				     0x0000000100000085, 0x00000000000001bf,
				     0x00000001000000b7, 0x0000000000010115,
				     0x0000000000001061, 0x0000000000000095 };
TEST_CASE("Test verifying ebpf programs with maps, without null check",
	  "[maps][null-check]")
{
	// bpf_map_lookup_elem -> 1
	set_available_helpers(std::vector<int32_t>{ 1 });
	set_map_descriptors(std::map<int, BpftimeMapDescriptor>{
		{ 2333, BpftimeMapDescriptor{ .original_fd = 2333,
					      .type = BPF_MAP_TYPE_HASH,
					      .key_size = 8,
					      .value_size = 4,
					      .max_entries = 8192,
					      .inner_map_fd = 0 } } });
	std::optional<std::string> ret;

	REQUIRE_NOTHROW([&]() {
		ret = verify_ebpf_program(prog_with_map_2,
					  std::size(prog_with_map_2),
					  "uprobe//proc/self/exe:uprobed_sub");
	}());

	if (ret.has_value()) {
		CAPTURE(ret.value());
	}
	REQUIRE_FALSE(ret.has_value());
}

TEST_CASE("Test verifying ebpf programs with maps, without definition",
	  "[maps][without-maps-and-helpers]")
{
	// bpf_map_lookup_elem -> 1
	set_available_helpers(std::vector<int32_t>{});
	set_map_descriptors(std::map<int, BpftimeMapDescriptor>{});
	std::optional<std::string> ret;

	REQUIRE_NOTHROW([&]() {
		ret = verify_ebpf_program(prog_with_map_2,
					  std::size(prog_with_map_2),
					  "uprobe//proc/self/exe:uprobed_sub");
	}());
	if (ret.has_value()) {
		CAPTURE(ret.value());
	}
	REQUIRE_THAT(ret.value(),
		     ContainsSubstring("6: invalid helper function id"));
}

TEST_CASE("Test verifying ebpf programs with maps, with helpers",
	  "[maps][without-maps-and-helpers]")
{
	// bpf_map_lookup_elem -> 1
	set_available_helpers(std::vector<int32_t>{ 1 });
	set_map_descriptors(std::map<int, BpftimeMapDescriptor>{});
	std::optional<std::string> ret;

	REQUIRE_NOTHROW([&]() {
		ret = verify_ebpf_program(prog_with_map_2,
					  std::size(prog_with_map_2),
					  "uprobe//proc/self/exe:uprobed_sub");
	}());
	if (ret.has_value()) {
		CAPTURE(ret.value());
	}
	REQUIRE_THAT(ret.value(),
		     ContainsSubstring("exit: Invalid map fd: 2333"));
}
