#include "catch2/catch_message.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <bpf_map/userspace/per_cpu_array_map.hpp>
#include <bpftime_shm_json.hpp>
#include <bpftime_shm_internal.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <sched.h>
#include <unistd.h>
#include <vector>
#include <bpf_map/map_common_def.hpp>
#include "common_def.hpp"
#include "catch2/internal/catch_run_context.hpp"
#include <algorithm>
#include <random>
#include <linux/bpf.h>
#include "bpf_attach_ctx.hpp"

using namespace boost::interprocess;
using namespace bpftime;

TEST_CASE("Test buffer_to_hex_string function")
{
	unsigned char buffer[] = { 0x12, 0x34, 0x56, 0x78 };
	std::string expected = "12345678";
	std::string result = buffer_to_hex_string(buffer, sizeof(buffer));
	REQUIRE(result == expected);
}

TEST_CASE("Test hex_string_to_buffer function")
{
	std::string hexString = "12345678";
	unsigned char expected[] = { 0x12, 0x34, 0x56, 0x78 };
	unsigned char buffer[sizeof(expected)];
	int result = hex_string_to_buffer(hexString, buffer, sizeof(buffer));
	REQUIRE(result == 0);
	REQUIRE(std::memcmp(buffer, expected, sizeof(buffer)) == 0);
}

TEST_CASE("Test hex_string_to_buffer function with invalid input")
{
	std::string hexString = "1234567";
	unsigned char buffer[4];
	int result = hex_string_to_buffer(hexString, buffer, sizeof(buffer));
	REQUIRE(result == -1);
}

static const char *SHM_NAME = "BPFTIME_TEST_SHM_JSON_IMPORT_EXPORT";
// original code from libebpf repo
const unsigned char bpf_add_mem_64_bit_minimal[] =
	""
	"\x61\x12\x00\x00\x00\x00\x00\x00"
	"\x61\x10\x04\x00\x00\x00\x00\x00"
	"\x0f\x20\x00\x00\x00\x00\x00\x00"
	"\x95\x00\x00\x00\x00\x00\x00\x00"
	"";

TEST_CASE("Test bpftime shm json import/export")
{
	bpftime_shm shm(SHM_NAME, shm_open_type::SHM_REMOVE_AND_CREATE);

	SECTION("Test shm json export")
	{
		// export empty shm to json
		REQUIRE(bpftime_export_shm_to_json(shm,
					   "/tmp/bpftime_test_shm_json.json") == 0);
		shm.add_bpf_prog(4,
				 (const ebpf_inst *)bpf_add_mem_64_bit_minimal,
				 4, "test_prog", BPF_PROG_TYPE_SOCKET_FILTER);
		shm.add_tracepoint(5, 123245, 6);
		bpf_link_create_args args = {
			.prog_fd = 4,
			.target_fd = 5,
		};
		shm.add_bpf_link(7, &args);
		shm.add_bpf_map(8, "test_map1",
				bpf_map_attr{ .type = BPF_MAP_TYPE_ARRAY,
					      .key_size = 4,
					      .value_size = 4,
					      .max_ents = 10 });
		shm.add_bpf_map(9, "test_map2",
				bpf_map_attr{ .type = BPF_MAP_TYPE_PERCPU_ARRAY,
					      .key_size = 4,
					      .value_size = 4,
					      .max_ents = 10 });
		shm.attach_perf_to_bpf(5, 4,{});
		int res = bpftime_export_shm_to_json(shm,
					   "/tmp/bpftime_test_shm_json.json");
        REQUIRE(res == 0);
	}

	SECTION("Test shm json import")
	{
		bpftime_shm shm2(SHM_NAME, shm_open_type::SHM_OPEN_ONLY);
		bpftime_import_shm_from_json(shm2,
		                 "/tmp/bpftime_test_shm_json.json");
		REQUIRE(shm2.is_prog_fd(4));
		REQUIRE(shm2.is_perf_fd(5));
		REQUIRE(shm2.is_exist_fake_fd(7));
		REQUIRE(shm2.is_map_fd(8));
		REQUIRE(shm2.is_map_fd(9));

		bpftime::bpf_attach_ctx ctx;
		ctx.init_attach_ctx_from_handlers(bpftime_get_agent_config());
	}
}
