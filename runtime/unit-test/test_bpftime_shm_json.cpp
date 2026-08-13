#include "catch2/catch_message.hpp"
#include "handler/link_handler.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <bpf_map/userspace/per_cpu_array_map.hpp>
#include <bpftime_shm_json.hpp>
#include <bpftime_shm_internal.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <fstream>
#include <json.hpp>
#include <sched.h>
#include <unistd.h>
#include <vector>
#include <bpf_map/map_common_def.hpp>
#include "common_def.hpp"
#include "catch2/internal/catch_run_context.hpp"
#include <algorithm>
#include <cerrno>
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

TEST_CASE("add_bpf_link rejects unsupported attach types")
{
	bpftime_shm shm("BPFTIME_TEST_SHM_UNSUPPORTED_LINK",
			shm_open_type::SHM_REMOVE_AND_CREATE);
	REQUIRE(shm.add_bpf_prog(
			4, (const ebpf_inst *)bpf_add_mem_64_bit_minimal, 4,
			"test_prog", BPF_PROG_TYPE_SOCKET_FILTER) >= 0);
	REQUIRE(shm.is_prog_fd(4));

	bpf_link_create_args args = { .prog_fd = 4,
				      .target_fd = 5,
				      .attach_type = BPF_TRACE_FENTRY };
	errno = 0;
	REQUIRE(shm.add_bpf_link(7, &args) == -1);
	REQUIRE(errno == EOPNOTSUPP);
	REQUIRE_FALSE(shm.is_exist_fake_fd(7));
}

TEST_CASE("add_bpf_link rejects malformed uprobe_multi args")
{
	bpftime_shm shm("BPFTIME_TEST_SHM_MALFORMED_UPROBE_MULTI_LINK",
			shm_open_type::SHM_REMOVE_AND_CREATE);
	REQUIRE(shm.add_bpf_prog(
			4, (const ebpf_inst *)bpf_add_mem_64_bit_minimal, 4,
			"test_prog", BPF_PROG_TYPE_SOCKET_FILTER) >= 0);
	const char path[] = "./victim";
	unsigned long offsets[] = { 0x123 };

	auto make_args = [&]() {
		bpf_link_create_args args{};
		args.prog_fd = 4;
		args.attach_type = bpftime::BPF_TRACE_UPROBE_MULTI;
		args.uprobe_multi.path = (uintptr_t)path;
		args.uprobe_multi.offsets = (uintptr_t)offsets;
		args.uprobe_multi.cnt = 1;
		return args;
	};

	auto args = make_args();
	args.uprobe_multi.cnt = 0;
	errno = 0;
	REQUIRE(shm.add_bpf_link(7, &args) == -1);
	REQUIRE(errno == EINVAL);

	args = make_args();
	args.uprobe_multi.path = 0;
	errno = 0;
	REQUIRE(shm.add_bpf_link(7, &args) == -1);
	REQUIRE(errno == EINVAL);

	args = make_args();
	args.uprobe_multi.offsets = 0;
	errno = 0;
	REQUIRE(shm.add_bpf_link(7, &args) == -1);
	REQUIRE(errno == EINVAL);

	args = make_args();
	args.uprobe_multi.flags = bpftime::BPF_F_UPROBE_MULTI_RETURN << 1;
	errno = 0;
	REQUIRE(shm.add_bpf_link(7, &args) == -1);
	REQUIRE(errno == EINVAL);
	REQUIRE_FALSE(shm.is_exist_fake_fd(7));
}

TEST_CASE("Test uprobe_multi link shm json import/export")
{
	bpftime_shm shm("BPFTIME_TEST_SHM_JSON_UPROBE_MULTI",
			shm_open_type::SHM_REMOVE_AND_CREATE);
	REQUIRE(shm.add_bpf_prog(
			4, (const ebpf_inst *)bpf_add_mem_64_bit_minimal, 4,
			"test_prog", BPF_PROG_TYPE_SOCKET_FILTER) >= 0);
	const char path[] = "./victim";
	unsigned long offsets[] = { 0x123, 0x456 };
	unsigned long ref_ctr_offsets[] = { 0, 0 };
	uint64_t cookies[] = { 7, 0 };
	bpf_link_create_args args{};
	args.prog_fd = 4;
	args.attach_type = bpftime::BPF_TRACE_UPROBE_MULTI;
	args.uprobe_multi.path = (uintptr_t)path;
	args.uprobe_multi.offsets = (uintptr_t)offsets;
	args.uprobe_multi.ref_ctr_offsets = (uintptr_t)ref_ctr_offsets;
	args.uprobe_multi.cookies = (uintptr_t)cookies;
	args.uprobe_multi.cnt = 2;
	args.uprobe_multi.flags = bpftime::BPF_F_UPROBE_MULTI_RETURN;
	args.uprobe_multi.pid = 123;
	REQUIRE(shm.add_bpf_link(7, &args) >= 0);

	const char *filename = "/tmp/bpftime_test_shm_json_uprobe_multi.json";
	REQUIRE(bpftime_export_shm_to_json(shm, filename) == 0);

	bpftime_shm shm2("BPFTIME_TEST_SHM_JSON_UPROBE_MULTI_IMPORT",
			 shm_open_type::SHM_REMOVE_AND_CREATE);
	REQUIRE(bpftime_import_shm_from_json(shm2, filename) == 0);
	REQUIRE(shm2.is_exist_fake_fd(7));

	const auto &handler = shm2.get_handler(7);
	REQUIRE(std::holds_alternative<bpf_link_handler>(handler));
	const auto &link = std::get<bpf_link_handler>(handler);
	REQUIRE(link.link_attach_type == bpftime::BPF_TRACE_UPROBE_MULTI);
	const auto &data = std::get<uprobe_multi_link_data>(link.data);
	REQUIRE(std::string(data.path.c_str()) == path);
	REQUIRE(data.flags == bpftime::BPF_F_UPROBE_MULTI_RETURN);
	REQUIRE(data.pid == 123);
	REQUIRE(data.entries.size() == 2);
	REQUIRE(data.entries[0].offset == offsets[0]);
	REQUIRE(data.entries[0].cookie == cookies[0]);
	REQUIRE(data.entries[1].offset == offsets[1]);
	REQUIRE_FALSE(data.entries[1].cookie.has_value());
}

TEST_CASE("bpf link json keeps legacy target_fd compatibility")
{
	bpftime_shm shm("BPFTIME_TEST_SHM_JSON_TARGET_FD_EXPORT",
			shm_open_type::SHM_REMOVE_AND_CREATE);
	REQUIRE(shm.add_bpf_prog(
			4, (const ebpf_inst *)bpf_add_mem_64_bit_minimal, 4,
			"test_prog", BPF_PROG_TYPE_SOCKET_FILTER) >= 0);
	REQUIRE(shm.add_tracepoint(5, 123245, 6) >= 0);
	bpf_link_create_args args = { .prog_fd = 4,
				      .target_fd = 5,
				      .attach_type = bpftime::BPF_PERF_EVENT };
	REQUIRE(shm.add_bpf_link(7, &args) >= 0);

	const char *exported_filename =
		"/tmp/bpftime_test_shm_json_target_fd_export.json";
	REQUIRE(bpftime_export_shm_to_json(shm, exported_filename) == 0);
	std::ifstream exported_file(exported_filename);
	auto exported = nlohmann::json::parse(exported_file);
	REQUIRE(exported["7"]["attr"]["target_fd"] == 5);
	REQUIRE(exported["7"]["attr"]["target_id"] == 5);

	const char *legacy_filename =
		"/tmp/bpftime_test_shm_json_legacy_link.json";
	std::ofstream legacy_file(legacy_filename);
	legacy_file << R"({
		"4": {
			"type": "bpf_prog_handler",
			"name": "test_prog",
			"attr": {
				"type": 0,
				"cnt": 4,
				"insns": "611200000000000061100400000000000f200000000000009500000000000000"
			}
		},
		"5": {
			"type": "bpf_perf_event_handler",
			"enabled": false,
			"attr": {
				"type": 2,
				"data_type": "tracepoint_perf_event_data",
				"pid": 123245,
				"tracepoint_id": 6
			}
		},
		"7": {
			"type": "bpf_link_handler",
			"attr": {
				"prog_fd": 4,
				"target_fd": 5
			}
		}
	})";
	legacy_file.close();

	bpftime_shm shm2("BPFTIME_TEST_SHM_JSON_TARGET_FD_IMPORT",
			 shm_open_type::SHM_REMOVE_AND_CREATE);
	REQUIRE(bpftime_import_shm_from_json(shm2, legacy_filename) == 0);
	REQUIRE(shm2.is_exist_fake_fd(7));
	const auto &handler = shm2.get_handler(7);
	REQUIRE(std::holds_alternative<bpf_link_handler>(handler));
	const auto &link = std::get<bpf_link_handler>(handler);
	REQUIRE(link.link_attach_type == bpftime::BPF_PERF_EVENT);
	REQUIRE(link.target_id == 5);
}

TEST_CASE("closing perf event does not delete uprobe_multi links")
{
	bpftime_shm shm("BPFTIME_TEST_SHM_UPROBE_MULTI_TARGET_LIFETIME",
			shm_open_type::SHM_REMOVE_AND_CREATE);
	REQUIRE(shm.add_bpf_prog(
			4, (const ebpf_inst *)bpf_add_mem_64_bit_minimal, 4,
			"test_prog", BPF_PROG_TYPE_SOCKET_FILTER) >= 0);
	REQUIRE(shm.add_tracepoint(5, 123245, 6) >= 0);
	const char path[] = "./victim";
	unsigned long offsets[] = { 0x123 };
	bpf_link_create_args args{};
	args.prog_fd = 4;
	args.target_fd = 5;
	args.attach_type = bpftime::BPF_TRACE_UPROBE_MULTI;
	args.uprobe_multi.path = (uintptr_t)path;
	args.uprobe_multi.offsets = (uintptr_t)offsets;
	args.uprobe_multi.cnt = 1;
	REQUIRE(shm.add_bpf_link(7, &args) >= 0);

	REQUIRE(shm.is_perf_fd(5));
	REQUIRE(shm.is_exist_fake_fd(7));
	shm.close_fd(5);
	REQUIRE_FALSE(shm.is_perf_fd(5));
	REQUIRE(shm.is_exist_fake_fd(7));
	const auto &handler = shm.get_handler(7);
	REQUIRE(std::holds_alternative<bpf_link_handler>(handler));
	const auto &link = std::get<bpf_link_handler>(handler);
	REQUIRE(link.link_attach_type == bpftime::BPF_TRACE_UPROBE_MULTI);
}

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
			.attach_type = bpftime::BPF_PERF_EVENT
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
		ctx.init_attach_ctx_from_handlers(bpftime_get_runtime_config());
	}
}
