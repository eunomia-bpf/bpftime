#include <catch2/catch_test_macros.hpp>
#include <bpftime-verifier.hpp>
#include <iterator>
using namespace bpftime;
using namespace verifier;

/*
       0:	b7 01 00 00 01 00 00 00	r1 = 1
       1:	85 00 00 00 a1 86 01 00	call 100001
       2:	95 00 00 00 00 00 00 00	exit
*/
const uint64_t simple_prog[] = { 0x00000001000001b7, 0x000186a100000085,
				 0x0000000000000095 };

TEST_CASE("Test whether verifying non-kernel helpers works", "[verify]")
{
	set_available_helpers(std::vector<int32_t>{ 100001 });
	set_non_kernel_helpers(
		{ { 100001,
		    BpftimeHelperProrotype{
			    .name = "my_helper",
			    .return_type = EBPF_RETURN_TYPE_INTEGER,
			    .argument_type = {
				    EBPF_ARGUMENT_TYPE_ANYTHING } } } });
	set_map_descriptors(std::map<int, BpftimeMapDescriptor>{});
	auto ret = verify_ebpf_program(simple_prog, std::size(simple_prog),
				       "uprobe/read");
	REQUIRE_FALSE(ret.has_value());
}
