#include <catch2/catch_message.hpp>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <catch2/catch_test_macros.hpp>
#include <bpftime-verifier.hpp>
using namespace bpftime;
using namespace verifier;
/*
a * b / 2 for 32 bit
clang -Xlinker --export-dynamic -O2 -target bpf -m32 -c example/bpf/mul.bpf.c -o prog.o
*/
static const unsigned char bpf_mul_optimized[] = { 0xb7, 0x00, 0x00, 0x00,
						   0x02, 0x00, 0x00, 0x00,
						   0x95, 0x00, 0x00, 0x00,
						   0x00, 0x00, 0x00, 0x00 };
static const uint64_t bad_ebpf_program[] = { 0xaabbccddeeff1122,
					     0x1122334455667788 };
TEST_CASE("Test verify simple ebpf programs", "[simple]")
{
	SECTION("verify correct programs")
	{
		auto ret = verify_ebpf_program(
			(uint64_t *)(uintptr_t)(&bpf_mul_optimized),
			sizeof(bpf_mul_optimized) / 8, "uprobe/read");
		REQUIRE_FALSE(ret.has_value());
	}
	SECTION("varify bad programs")
	{
		auto ret = verify_ebpf_program(bad_ebpf_program,
					       sizeof(bad_ebpf_program),
					       "uprobe/read");
		INFO(ret.value());
		REQUIRE(ret.has_value());
	}
}
