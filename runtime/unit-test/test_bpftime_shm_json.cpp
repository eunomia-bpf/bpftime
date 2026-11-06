#include <bpftime_shm_json.hpp>
#include <bpftime_shm_internal.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_message.hpp>
#include <json.hpp>
#include <linux/bpf.h>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <unistd.h>
#include "common_def.hpp"

using namespace bpftime;
using json = nlohmann::json;

#ifdef __linux__
namespace
{
thread_local bpftime_shm *close_mock_shm = nullptr;
struct close_mock_guard {
	bpftime_shm *previous = close_mock_shm;
	explicit close_mock_guard(bpftime_shm &shm) { close_mock_shm = &shm; }
	~close_mock_guard() { close_mock_shm = previous; }
};
} // namespace

extern "C" int __real_close(int fd);
extern "C" int __wrap_close(int fd)
{
	// Model syscall_context::handle_close only within this regression.
	if (close_mock_shm) close_mock_shm->close_fd(fd);
	return __real_close(fd);
}
#endif

namespace
{
struct json_fixture {
	static unsigned next_id;
	std::string name = "bpftime-json-" + std::to_string(getpid()) + "-" +
			   std::to_string(++next_id);
	::shm_remove remover{ std::string(name) };
	bpftime_shm shm{ name.c_str(), shm_open_type::SHM_REMOVE_AND_CREATE };
	std::string directory;
	std::string path;

	json_fixture()
	{
		char pattern[] = "/tmp/bpftime-json-XXXXXX";
		auto result = mkdtemp(pattern);
		REQUIRE(result != nullptr);
		directory = result;
		path = directory + "/state.json";
	}
	~json_fixture()
	{
		std::error_code error;
		std::filesystem::remove_all(directory, error);
	}
	void write(const std::string &value)
	{
		std::ofstream file(path);
		file << value;
		file.close();
		REQUIRE(file.good());
	}
	std::string read() const
	{
		std::ifstream file(path);
		REQUIRE(file.is_open());
		return { std::istreambuf_iterator<char>(file), {} };
	}
	json snapshot()
	{
		REQUIRE(bpftime_export_shm_to_json(shm, path.c_str()) == 0);
		return json::parse(read());
	}
};
unsigned json_fixture::next_id = 0;

struct map_ops_guard {
	int type;
	bpftime_map_ops previous;
	explicit map_ops_guard(int type)
		: type(type), previous(global_map_ops_table[type]) {}
	~map_ops_guard() { global_map_ops_table[type] = previous; }
};

void allow_custom_maps(bpftime_shm &shm, bool enabled = true)
{
	auto config = shm.get_runtime_config();
	config.allow_non_buildin_map_types = enabled;
	shm.set_runtime_config(std::move(config));
}

// r0 = 0; exit. No relocation, helper, or attachment is needed to encode it.
const ebpf_inst instructions[] = {
	{ .code = 0xb7, .dst_reg = 0, .src_reg = 0, .off = 0, .imm = 0 },
	{ .code = 0x95, .dst_reg = 0, .src_reg = 0, .off = 0, .imm = 0 },
};

void add_program(bpftime_shm &shm, int fd)
{
	REQUIRE(shm.add_bpf_prog(fd, instructions, 2, "json_program",
				 BPF_PROG_TYPE_SOCKET_FILTER) == fd);
	CHECK(static_cast<int>(std::get<bpf_prog_handler>(shm.get_manager()->get_handler(fd)).type) ==
	      BPF_PROG_TYPE_SOCKET_FILTER);
}
void add_map(bpftime_shm &shm, int fd)
{
	REQUIRE(shm.add_bpf_map(fd, "json_map",
			       { .type = BPF_MAP_TYPE_ARRAY, .key_size = 4,
				 .value_size = 8, .max_ents = 1 }) == fd);
}
} // namespace

TEST_CASE("JSON export destination can resemble its temporary filename",
	  "[json][json-temp-collision]")
{
	// This is intentionally the first JSON test. It is also run separately
	// with [json-temp-collision], ensuring the per-process sequence starts at
	// zero and the first temporary candidate equals the requested basename.
	json_fixture fixture;
	fixture.path = fixture.directory + "/.bpftime-json-" +
		       std::to_string(getpid()) + "-0";
	REQUIRE(bpftime_export_shm_to_json(fixture.shm, fixture.path.c_str()) == 0);
	CHECK(json::parse(fixture.read()) == json::object());
}

#ifdef __linux__
TEST_CASE("JSON export closes OS descriptors without deleting handlers", "[json]")
{
	json_fixture fixture;
	// Discover the two slots export will use, without reserving OS descriptors
	// for the shared handlers. This models imported definition-only IDs.
	const int directory_fd = open(fixture.directory.c_str(), O_RDONLY | O_DIRECTORY);
	REQUIRE(directory_fd >= 0);
	const int file_fd = open(fixture.directory.c_str(), O_RDONLY | O_DIRECTORY);
	REQUIRE(file_fd >= 0);
	REQUIRE(__real_close(directory_fd) == 0);
	REQUIRE(__real_close(file_fd) == 0);
	add_map(fixture.shm, directory_fd);
	add_map(fixture.shm, file_fd);
	const auto original = fixture.snapshot();
	std::string destination = fixture.path;
	int expected_result = 0;
	SECTION("successful export closes its temporary file and directory") {}
	SECTION("rejected destination still closes its directory")
	{
		destination = fixture.directory;
		expected_result = -1;
	}
	int result;
	{
		close_mock_guard guard(fixture.shm);
		result = bpftime_export_shm_to_json(fixture.shm, destination.c_str());
	}
	CHECK(result == expected_result);
	for (int fd : { directory_fd, file_fd }) {
		errno = 0;
		CHECK(fcntl(fd, F_GETFD) == -1);
		CHECK(errno == EBADF);
	}
	CHECK(fixture.shm.get_manager()->is_allocated(directory_fd));
	CHECK(fixture.shm.get_manager()->is_allocated(file_fd));
	CHECK(fixture.snapshot() == original);
}
#endif

TEST_CASE("JSON byte encoding is strict and never throws", "[json]")
{
	unsigned char expected[] = { 0x12, 0x34, 0xab, 0xff };
	CHECK(buffer_to_hex_string(expected, sizeof(expected)) == "1234abff");
	unsigned char actual[4]{};
	REQUIRE(hex_string_to_buffer("1234ABff", actual, sizeof(actual)) == 0);
	CHECK(std::memcmp(actual, expected, sizeof(actual)) == 0);
	for (const auto *bad : { "1234567", "123456789", "zz345678", "1z345678",
				"-1345678", "+1345678", " 1345678" }) {
		CAPTURE(bad);
		int result = 0;
		REQUIRE_NOTHROW(result = hex_string_to_buffer(bad, actual, sizeof(actual)));
		CHECK(result < 0);
	}
}

TEST_CASE("JSON definition graph round-trips typed runtime metadata", "[json]")
{
	json_fixture source;
	json_fixture restored;
	add_program(source.shm, 20);
	add_map(source.shm, 8);
	REQUIRE(source.shm.add_tracepoint(30, -1, 1) == 30);
	const uint64_t cookie = (uint64_t{ 1 } << 61) + 17;
	bpf_link_create_args args{};
	args.prog_fd = 20;
	args.target_fd = 30;
	args.attach_type = BPFTIME_BPF_PERF_EVENT_ATTACH_TYPE;
	args.perf_event.bpf_cookie = cookie;
	REQUIRE(source.shm.add_bpf_link(10, &args) == 10);
	args.perf_event.bpf_cookie = 0;
	REQUIRE(source.shm.add_bpf_link(11, &args) == 11);
	const uint64_t offset = (uint64_t{ 1 } << 40) + 17;
	const size_t ref_counter = size_t{ 1 } << 33;
	REQUIRE(source.shm.add_uprobe(50, -1, "/not/attached/by/import", offset,
				     false, ref_counter) == 50);
	REQUIRE(source.shm.add_uprobe(51, -1, "/not/attached/by/import", offset,
				     true, ref_counter) == 51);
	REQUIRE(source.shm.add_kprobe(60, "do_sys_open", offset, false, ref_counter) == 60);
	REQUIRE(source.shm.add_kprobe(61, "do_sys_open", offset, true, ref_counter) == 61);
	REQUIRE(source.shm.add_software_perf_event(40, 0, 1024, 10) == 40);
	REQUIRE(source.shm.perf_event_enable(40) == 0);
	REQUIRE(source.shm.add_uprobe_override(52, -1, "/not/attached/by/import",
					      offset, true) == 52);
	REQUIRE(source.shm.perf_event_disable(52) == 0);
	REQUIRE(source.shm.add_uprobe_override(53, -1, "/not/attached/by/import",
					      offset, false) == 53);
	REQUIRE(source.shm.perf_event_disable(53) == 0);

	const auto first = source.snapshot();
	REQUIRE(bpftime_import_shm_from_json(restored.shm, source.path.c_str()) == 0);
	CHECK(restored.snapshot() == first);
	const auto *manager = restored.shm.get_manager();
	const auto &link = std::get<bpf_link_handler>(manager->get_handler(10));
	CHECK(link.args.attach_type == BPFTIME_BPF_PERF_EVENT_ATTACH_TYPE);
	REQUIRE(link.attach_cookie.has_value());
	CHECK(*link.attach_cookie == cookie);
	const auto &zero_link = std::get<bpf_link_handler>(manager->get_handler(11));
	REQUIRE(zero_link.attach_cookie.has_value());
	CHECK(*zero_link.attach_cookie == 0);
	const auto &uprobe = std::get<uprobe_perf_event_data>(
		std::get<bpf_perf_event_handler>(manager->get_handler(50)).data);
	CHECK(uprobe.offset == offset);
	CHECK(uprobe.ref_ctr_off == ref_counter);
	CHECK_FALSE(std::get<bpf_perf_event_handler>(manager->get_handler(52)).enabled);
	CHECK_FALSE(std::get<bpf_perf_event_handler>(manager->get_handler(53)).enabled);
	const auto &prog = std::get<bpf_prog_handler>(manager->get_handler(20));
	CHECK(static_cast<int>(prog.type) == BPF_PROG_TYPE_SOCKET_FILTER);
	REQUIRE(prog.insns.size() == 2);
	CHECK(std::memcmp(prog.insns.data(), instructions, sizeof(instructions)) == 0);
}

TEST_CASE("JSON legacy links avoid standard and reserved descriptors", "[json]")
{
	json_fixture source;
	json_fixture restored;
	add_program(source.shm, 20);
	REQUIRE(source.shm.add_tracepoint(30, -1, 1) == 30);
	auto legacy = source.snapshot();
	legacy["20"]["attr"]["attach_fds"] = { 30, 30 };
	int link_fd = STDERR_FILENO + 1;
	SECTION("standard descriptors are not synthesized") {}
	SECTION("existing and explicitly reserved descriptors are skipped")
	{
		add_map(restored.shm, 3);
		add_map(source.shm, 4);
		legacy["4"] = source.snapshot()["4"];
		link_fd = 5;
	}
	source.write(legacy.dump());
	REQUIRE(bpftime_import_shm_from_json(restored.shm, source.path.c_str()) == 0);
	const auto *manager = restored.shm.get_manager();
	REQUIRE(std::holds_alternative<bpf_link_handler>(manager->get_handler(link_fd)));
	const auto &link = std::get<bpf_link_handler>(manager->get_handler(link_fd));
	CHECK(link.prog_id == 20);
	CHECK(link.attach_target_id == 30);
	const auto normalized = restored.snapshot();
	size_t link_count = 0;
	for (const auto &record : normalized)
		if (record.at("type") == "bpf_link_handler") ++link_count;
	CHECK(link_count == 1);
	for (int fd = STDIN_FILENO; fd <= STDERR_FILENO; ++fd) {
		CHECK_FALSE(manager->is_allocated(fd));
		// Exercise the intercepted close path without closing the test's stdio.
		restored.shm.close_fd(fd);
	}
	CHECK(restored.snapshot() == normalized);
}

TEST_CASE("JSON import failures preserve existing objects and references", "[json]")
{
	json_fixture fixture;
	add_program(fixture.shm, 20);
	add_map(fixture.shm, 8);
	REQUIRE(fixture.shm.add_tracepoint(30, -1, 1) == 30);
	bpf_link_create_args args{};
	args.prog_fd = 20;
	args.target_fd = 30;
	args.attach_type = BPFTIME_BPF_PERF_EVENT_ATTACH_TYPE;
	REQUIRE(fixture.shm.add_bpf_link(10, &args) == 10);
	const auto original = fixture.snapshot();
	for (int invalid_fd : { -1, static_cast<int>(fixture.shm.get_manager()->size()) }) {
		int result = 0;
		REQUIRE_NOTHROW(result = bpftime::bpftime_import_shm_handler_from_json(
				       fixture.shm, invalid_fd, original["8"].dump().c_str()));
		CHECK(result < 0);
		CHECK(fixture.snapshot() == original);
	}
	CHECK(bpftime::bpftime_import_shm_handler_from_json(fixture.shm, 100, nullptr) < 0);
	CHECK(bpftime_import_shm_from_json(fixture.shm, nullptr) < 0);
	CHECK(bpftime_export_shm_to_json(fixture.shm, nullptr) < 0);
	const uint32_t key = 0;
	const uint64_t value = 0x123456789abcdef0;
	REQUIRE(fixture.shm.bpf_map_update_elem(8, &key, &value, 0, false) == 0);

	for (const auto *bad : { "", "{", "[]", "null", "0", "{}",
				"{\"type\":\"unknown_handler\"}" }) {
		CAPTURE(bad);
		int result = 0;
		REQUIRE_NOTHROW(result = bpftime::bpftime_import_shm_handler_from_json(
				fixture.shm, 100, bad));
		CHECK(result < 0);
		CHECK(fixture.snapshot() == original);
	}
	// Replacing a link with a map used to cascade deletion to the old perf
	// target. A collision must be rejected before any graph mutation.
	for (int fd : { 8, 10, 20, 30 }) {
		CAPTURE(fd);
		CHECK(bpftime::bpftime_import_shm_handler_from_json(
			      fixture.shm, fd, original["8"].dump().c_str()) < 0);
		CHECK(fixture.snapshot() == original);
	}
	json invalid = { { "1", original["8"] },
			 { "100", { { "type", "unknown_handler" } } } };
	fixture.write(invalid.dump());
	int result = 0;
	REQUIRE_NOTHROW(result = bpftime_import_shm_from_json(fixture.shm, fixture.path.c_str()));
	CHECK(result < 0);
	CHECK(fixture.snapshot() == original);
	// Unique staging IDs come from the decoded map, not publication-time
	// deduplication. Reject repeated keys and noncanonical numeric aliases.
	for (const auto *second_key : { "100", "0100" }) {
		const auto record = original["8"].dump();
		fixture.write(R"({"100":)" + record + R"(,")" +
			      second_key + R"(":)" + record + "}");
		CHECK(bpftime_import_shm_from_json(fixture.shm, fixture.path.c_str()) < 0);
		CHECK(fixture.snapshot() == original);
	}
	const auto *restored_value = fixture.shm.bpf_map_lookup_elem(8, &key, false);
	REQUIRE(restored_value != nullptr);
	CHECK(*static_cast<const uint64_t *>(restored_value) == value);
}

TEST_CASE("JSON custom perf payloads may reuse built-in numeric types", "[json]")
{
	for (const int type : {
		     static_cast<int>(bpftime::bpf_event_type::BPF_TYPE_UPROBE),
		     static_cast<int>(bpftime::bpf_event_type::BPF_TYPE_KPROBE),
		     static_cast<int>(bpftime::bpf_event_type::PERF_TYPE_TRACEPOINT),
		     static_cast<int>(bpftime::bpf_event_type::PERF_TYPE_SOFTWARE),
		     -1, 123456 }) {
		CAPTURE(type);
		json_fixture source;
		json_fixture restored;
		const json record = {
			{ "type", "bpf_perf_event_handler" },
			{ "enabled", false },
			{ "attr", { { "type", type },
				    { "data_type", "custom_perf_event_data" },
				    { "attach_argument", "target=custom" } } }
		};
		REQUIRE(bpftime::bpftime_import_shm_handler_from_json(
				source.shm, 40, record.dump().c_str()) == 0);
		const auto &handler = std::get<bpf_perf_event_handler>(
			source.shm.get_manager()->get_handler(40));
		CHECK(handler.type == type);
		CHECK_FALSE(handler.enabled);
		REQUIRE(std::holds_alternative<custom_perf_event_data>(handler.data));
		CHECK(source.snapshot()["40"] == record);
		REQUIRE(bpftime_import_shm_from_json(
				restored.shm, source.path.c_str()) == 0);
		CHECK(restored.snapshot()["40"] == record);
	}
}

TEST_CASE("JSON custom perf payloads require explicit complete metadata", "[json]")
{
	json_fixture fixture;
	json record = {
		{ "type", "bpf_perf_event_handler" },
		{ "enabled", true },
		{ "attr", { { "type", 123456 },
			    { "data_type", "custom_perf_event_data" },
			    { "attach_argument", "target=custom" } } }
	};
	SECTION("unknown legacy type cannot infer a custom payload")
	{
		record["attr"].erase("data_type");
	}
	SECTION("custom tag does not supply a missing argument")
	{
		record["attr"].erase("attach_argument");
	}
	SECTION("unknown payload discriminator is rejected")
	{
		record["attr"]["data_type"] = "unknown_perf_event_data";
	}
	CHECK(bpftime::bpftime_import_shm_handler_from_json(
		      fixture.shm, 40, record.dump().c_str()) < 0);
	CHECK_FALSE(fixture.shm.get_manager()->is_allocated(40));
}

TEST_CASE("JSON callback-only maps never initialize or release external resources", "[json]")
{
	constexpr int custom_type = 2046;
	static thread_local unsigned alloc_calls, free_calls, lookup_calls;
	static thread_local uint64_t value;
	alloc_calls = free_calls = lookup_calls = 0;
	value = 123;
	auto allocate = +[](int, const char *, bpf_map_attr) {
		++alloc_calls;
		return 0;
	};
	auto release = +[](int) { ++free_calls; };
	bpftime_map_ops ops{};
	ops.elem_lookup = +[](int, const void *, bool) -> void * {
		++lookup_calls;
		return &value;
	};
	map_ops_guard callbacks(custom_type);
	REQUIRE(bpftime_register_map_ops(custom_type, &ops) == 0);
	json_fixture fixture;
	allow_custom_maps(fixture.shm);
	add_map(fixture.shm, 8);
	const auto original = fixture.snapshot();
	auto record = original["8"];
	record["attr"]["map_type"] = custom_type;

	SECTION("metadata round-trips without invoking callbacks")
	{
		REQUIRE(bpftime::bpftime_import_shm_handler_from_json(
				fixture.shm, 40, record.dump().c_str()) == 0);
		const auto &map = std::get<bpf_map_handler>(
			fixture.shm.get_manager()->get_handler(40));
		CHECK_FALSE(map.has_map_impl());
		const auto document = fixture.snapshot();
		CHECK(document["40"] == record);
		json_fixture restored;
		allow_custom_maps(restored.shm);
		REQUIRE(bpftime_import_shm_from_json(restored.shm, fixture.path.c_str()) == 0);
		CHECK(restored.snapshot() == document);
		CHECK(lookup_calls == 0);
		const uint32_t key = 0;
		CHECK(restored.shm.bpf_map_lookup_elem(40, &key, false) == &value);
		CHECK(lookup_calls == 1);
		// Even later registration of lifecycle callbacks cannot make a
		// metadata-only handler release resources that import never owned.
		ops.alloc_map = allocate;
		ops.map_free = release;
		REQUIRE(bpftime_register_map_ops(custom_type, &ops) == 0);
		fixture.shm.close_fd(40);
		restored.shm.close_fd(40);
	}
	SECTION("a later allocation failure rolls back staged metadata")
	{
		boost::interprocess::managed_shared_memory memory(
			boost::interprocess::open_only, fixture.name.c_str());
		const auto free_before = memory.get_free_memory();
		auto oversized = original["8"];
		oversized["attr"]["max_entries"] = memory.get_size() / 8 + 1;
		fixture.write(json{ { "1", record }, { "2", oversized } }.dump());
		CHECK(bpftime_import_shm_from_json(fixture.shm, fixture.path.c_str()) < 0);
		CHECK(memory.get_free_memory() == free_before);
		CHECK(fixture.snapshot() == original);
		CHECK(lookup_calls == 0);
	}
	SECTION("ineligible definitions are rejected before callbacks")
	{
		SECTION("custom types must be enabled") { allow_custom_maps(fixture.shm, false); }
		SECTION("data callbacks must be registered") { ops = {}; }
		SECTION("external initialization is not transactional") { ops.alloc_map = allocate; }
		SECTION("external cleanup is not owned by import") { ops.map_free = release; }
		SECTION("negative types do not index the callback table") { record["attr"]["map_type"] = -1; }
		SECTION("oversized types do not index the callback table") {
			record["attr"]["map_type"] = static_cast<int>(bpftime::bpf_map_type::BPF_MAP_TYPE_MAX);
		}
		REQUIRE(bpftime_register_map_ops(custom_type, &ops) == 0);
		errno = 0;
		CHECK(bpftime::bpftime_import_shm_handler_from_json(
			      fixture.shm, 40, record.dump().c_str()) < 0);
		CHECK(errno == EINVAL);
		CHECK(fixture.snapshot() == original);
		CHECK(lookup_calls == 0);
	}
#ifndef BPFTIME_ENABLE_CUDA_ATTACH
	SECTION("unavailable native maps cannot become callback-only maps")
	{
		constexpr int native_type =
			static_cast<int>(bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP);
		map_ops_guard native_callbacks(native_type);
		REQUIRE(bpftime_register_map_ops(native_type, &ops) == 0);
		record["attr"]["map_type"] = native_type;
		CHECK(bpftime::bpftime_import_shm_handler_from_json(
			      fixture.shm, 40, record.dump().c_str()) < 0);
		CHECK(fixture.snapshot() == original);
	}
#endif
	CHECK(alloc_calls == 0);
	CHECK(free_calls == 0);
}

TEST_CASE("JSON export preserves externally backed map definitions", "[json]")
{
	json_fixture source;
	json_fixture restored;
	add_map(source.shm, 8);
	auto &map = const_cast<bpf_map_handler &>(
		std::get<bpf_map_handler>(source.shm.get_manager()->get_handler(8)));
	// Exercise metadata export only. Keep the backing implementation/type
	// as an ordinary array, so this test needs no kernel, GPU, or callbacks.
	for (const auto type : {
		     bpftime::bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERF_EVENT_ARRAY,
		     bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_HASH_MAP,
		     bpftime::bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP,
		     bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_MAP,
		     bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_RINGBUF_MAP,
		     bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP,
#ifndef BPFTIME_ENABLE_CUDA_ATTACH
		     bpftime::bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP,
		     bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP,
#endif
		     static_cast<bpftime::bpf_map_type>(2047) }) {
		CAPTURE(static_cast<int>(type));
		map.attr.type = static_cast<int>(type);
		map.attr.kernel_bpf_map_id = 123;
		map.attr.gpu_thread_count = 4096;
		const auto document = source.snapshot();
		CHECK(document["8"]["attr"]["map_type"] == static_cast<int>(type));
		CHECK(document["8"]["attr"]["kernel_bpf_map_id"] == 123);
		CHECK(document["8"]["attr"]["gpu_thread_count"] == 4096);
		// Restoring external resources remains a separate capability:
		// failure must not prevent exporting their definitions.
		CHECK(bpftime_import_shm_from_json(restored.shm, source.path.c_str()) < 0);
		CHECK_FALSE(restored.shm.get_manager()->is_allocated(8));
	}
}

#ifdef BPFTIME_ENABLE_CUDA_ATTACH
TEST_CASE("JSON host GPU arrays preserve host storage and metadata", "[json]")
{
	for (const auto type : {
		     bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP,
		     bpftime::bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP }) {
		CAPTURE(static_cast<int>(type));
		json_fixture source;
		json_fixture restored;
		add_map(source.shm, 8);
		auto record = source.snapshot()["8"];
		record["attr"]["map_type"] = static_cast<int>(type);
		record["attr"]["gpu_thread_count"] = 2;
		REQUIRE(bpftime::bpftime_import_shm_handler_from_json(
				restored.shm, 8, record.dump().c_str()) == 0);
		CHECK(restored.snapshot()["8"] == record);
		const auto &map = std::get<bpf_map_handler>(
			restored.shm.get_manager()->get_handler(8));
		const uint32_t expected_size =
			type == bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP ? 8 : 16;
		CHECK(map.get_userspace_value_size() == expected_size);
		const uint32_t key = 0;
		const auto *value = static_cast<const unsigned char *>(
			restored.shm.bpf_map_lookup_elem(8, &key, false));
		REQUIRE(value != nullptr);
		for (uint32_t i = 0; i != expected_size; ++i)
			CHECK(value[i] == 0);

		// A later allocation failure must release the prepared host array.
		boost::interprocess::managed_shared_memory memory(
			boost::interprocess::open_only, restored.name.c_str());
		const auto free_before = memory.get_free_memory();
		const auto original = restored.snapshot();
		auto oversized = source.snapshot()["8"];
		oversized["attr"]["max_entries"] = memory.get_size() / 8 + 1;
		source.write(json{ { "1", record }, { "2", oversized } }.dump());
		CHECK(bpftime_import_shm_from_json(restored.shm, source.path.c_str()) < 0);
		CHECK(memory.get_free_memory() == free_before);
		CHECK(restored.snapshot() == original);
	}
}

TEST_CASE("JSON host GPU arrays reject inconsistent dimensions", "[json]")
{
	json_fixture source;
	json_fixture restored;
	add_map(source.shm, 8);
	auto record = source.snapshot()["8"];
	record["attr"]["map_type"] =
		static_cast<int>(bpftime::bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP);
	SECTION("thread count cannot silently default to one")
	{
		record["attr"]["gpu_thread_count"] = 0;
	}
	SECTION("per-entry size cannot overflow the public uint32 size")
	{
		record["attr"]["gpu_thread_count"] = uint64_t{1} << 32;
	}
	SECTION("per-thread array keys must be four bytes")
	{
		record["attr"]["key_size"] = 8;
	}
	SECTION("shared array keys must be four bytes")
	{
		record["attr"]["map_type"] =
			static_cast<int>(bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP);
		record["attr"]["key_size"] = 8;
	}
	errno = 0;
	CHECK(bpftime::bpftime_import_shm_handler_from_json(
		      restored.shm, 8, record.dump().c_str()) < 0);
	CHECK(errno == EINVAL);
	CHECK_FALSE(restored.shm.get_manager()->is_allocated(8));
}
#endif

TEST_CASE("JSON sparse map dimensions do not imply dense allocation", "[json]")
{
	json_fixture source;
	json_fixture restored;
	add_map(source.shm, 8);
	auto record = source.snapshot()["8"];
	SECTION("Bloom filter stores bits rather than whole values")
	{
		record["attr"]["map_type"] = BPF_MAP_TYPE_BLOOM_FILTER;
		record["attr"]["key_size"] = 0;
		record["attr"]["value_size"] = 4096;
		record["attr"]["max_entries"] = 1048576;
	}
	SECTION("LPM trie does not preallocate every entry")
	{
		record["attr"]["map_type"] = BPF_MAP_TYPE_LPM_TRIE;
		record["attr"]["key_size"] = 8;
		record["attr"]["value_size"] = 4096;
		record["attr"]["max_entries"] = std::numeric_limits<uint32_t>::max();
	}
	SECTION("Per-CPU hash only allocates its initial templates")
	{
		record["attr"]["map_type"] = BPF_MAP_TYPE_PERCPU_HASH;
		record["attr"]["max_entries"] = std::numeric_limits<uint32_t>::max();
	}
#ifdef BPFTIME_BUILD_WITH_LIBBPF
	SECTION("Kernel array with an ID is lazy")
	{
		record["attr"]["map_type"] =
			static_cast<int>(bpftime::bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY);
		record["attr"]["value_size"] = 4096;
		record["attr"]["max_entries"] = std::numeric_limits<uint32_t>::max();
		record["attr"]["kernel_bpf_map_id"] = 123;
	}
#endif
	REQUIRE(bpftime::bpftime_import_shm_handler_from_json(
			restored.shm, 8, record.dump().c_str()) == 0);
	CHECK(restored.snapshot()["8"] == record);
}

TEST_CASE("JSON dense map dimensions reject uint32 multiplication overflow", "[json]")
{
	json_fixture source;
	json_fixture restored;
	add_map(source.shm, 8);
	auto record = source.snapshot()["8"];
	for (const int type : { BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_QUEUE, BPF_MAP_TYPE_STACK }) {
		CAPTURE(type);
		record["attr"]["map_type"] = type;
		record["attr"]["key_size"] = type == BPF_MAP_TYPE_ARRAY ? 4 : 0;
		record["attr"]["value_size"] = 65536;
		record["attr"]["max_entries"] = 65536;
		errno = 0;
		CHECK(bpftime::bpftime_import_shm_handler_from_json(
			      restored.shm, 8, record.dump().c_str()) < 0);
		CHECK(errno == EINVAL);
		CHECK_FALSE(restored.shm.get_manager()->is_allocated(8));
	}
}

TEST_CASE("JSON rejects map IDs that narrow in kernel wrappers", "[json]")
{
#ifdef BPFTIME_BUILD_WITH_LIBBPF
	json_fixture source;
	json_fixture restored;
	add_map(source.shm, 8);
	auto record = source.snapshot()["8"];
	for (const auto type : {
		     bpftime::bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY,
		     bpftime::bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_HASH,
		     bpftime::bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERCPU_ARRAY }) {
		record["attr"]["map_type"] = static_cast<int>(type);
		record["attr"]["kernel_bpf_map_id"] = std::numeric_limits<uint32_t>::max();
		errno = 0;
		CHECK(bpftime::bpftime_import_shm_handler_from_json(
			      restored.shm, 8, record.dump().c_str()) < 0);
		CHECK(errno == EINVAL);
		CHECK_FALSE(restored.shm.get_manager()->is_allocated(8));
	}
#endif
}

TEST_CASE("JSON rejects fixed hash bucket rounding beyond uint32", "[json]")
{
#ifndef BPFTIME_USE_VAR_HASH_MAP
	json_fixture source;
	json_fixture restored;
	add_map(source.shm, 8);
	auto record = source.snapshot()["8"];
	record["attr"]["map_type"] = BPF_MAP_TYPE_HASH;
	record["attr"]["max_entries"] = std::numeric_limits<uint32_t>::max();
	errno = 0;
	CHECK(bpftime::bpftime_import_shm_handler_from_json(
		      restored.shm, 8, record.dump().c_str()) < 0);
	CHECK(errno == EINVAL);
	CHECK_FALSE(restored.shm.get_manager()->is_allocated(8));
#endif
}

TEST_CASE("JSON export directly validates link endpoint types", "[json]")
{
	json_fixture fixture;
	add_program(fixture.shm, 20);
	REQUIRE(fixture.shm.add_tracepoint(30, -1, 1) == 30);
	bpf_link_create_args args{};
	args.prog_fd = 20;
	args.target_fd = 30;
	REQUIRE(fixture.shm.add_bpf_link(10, &args) == 10);
	const auto original = fixture.snapshot();
	auto &link = const_cast<bpf_link_handler &>(
		std::get<bpf_link_handler>(fixture.shm.get_manager()->get_handler(10)));
	for (int target : { -1, 20, static_cast<int>(fixture.shm.get_manager()->size()) }) {
		CAPTURE(target);
		link.attach_target_id = target;
		CHECK(bpftime_export_shm_to_json(fixture.shm, fixture.path.c_str()) < 0);
		CHECK(json::parse(fixture.read()) == original);
	}
	link.attach_target_id = 30;
	CHECK(fixture.snapshot() == original);
}

TEST_CASE("JSON resource preparation failures release unpublished maps", "[json]")
{
	json_fixture fixture;
	add_map(fixture.shm, 8);
	const auto original = fixture.snapshot();
	boost::interprocess::managed_shared_memory observer(
		boost::interprocess::open_only, fixture.name.c_str());
	const auto free_before = observer.get_free_memory();
	auto oversized_map = original["8"];
	// The dimensions fit uint32_t, but the allocation cannot fit this
	// segment. No backing pages need to be touched to discover that.
	REQUIRE(observer.get_size() / 8 + 1 < std::numeric_limits<uint32_t>::max());
	oversized_map["attr"]["max_entries"] = observer.get_size() / 8 + 1;
	const json input = { { "1", original["8"] }, { "2", oversized_map } };
	for (int attempt = 0; attempt < 3; ++attempt) {
		fixture.write(input.dump());
		int result = 0;
		REQUIRE_NOTHROW(result = bpftime_import_shm_from_json(
				       fixture.shm, fixture.path.c_str()));
		CHECK(result < 0);
		CHECK(fixture.snapshot() == original);
		CHECK(observer.get_free_memory() == free_before);
	}
}

TEST_CASE("JSON rollback frees a map when its refcount allocation fails", "[json]")
{
	json_fixture fixture;
	add_map(fixture.shm, 8);
	const auto original = fixture.snapshot();
	const auto payload = original["8"].dump();
	const auto attr = std::get<bpf_map_handler>(
		fixture.shm.get_manager()->get_handler(8)).attr;
	boost::interprocess::managed_shared_memory observer(
		boost::interprocess::open_only, fixture.name.c_str());
	const auto free_before = observer.get_free_memory();
	bool reached_refcount_failure = false;
	for (size_t available = 64; available <= 4096; available += 16) {
		REQUIRE(free_before > available);
		void *allocation = nullptr;
		try {
			allocation = observer.allocate(free_before - available);
		} catch (const boost::interprocess::bad_alloc &) {
			continue;
		}
		struct reservation_guard {
			boost::interprocess::managed_shared_memory &memory;
			void *allocation;
			~reservation_guard() { memory.deallocate(allocation); }
		} reservation{ observer, allocation };
		const auto constrained_free = observer.get_free_memory();
		// Calibrate against this allocator instead of assuming Boost's block
		// size. has_map_impl && !can_share_map_impl proves the failure occurs
		// after the map implementation succeeds, at its refcount allocation.
		{
			bpf_map_handler probe(1, "json_map", observer, attr);
			try {
				probe.map_init(observer);
			} catch (const boost::interprocess::bad_alloc &) {
				reached_refcount_failure = probe.has_map_impl() &&
							   !probe.can_share_map_impl();
			}
			probe.map_free(observer);
		}
		REQUIRE(observer.get_free_memory() == constrained_free);
		if (!reached_refcount_failure)
			continue;
		for (int attempt = 0; attempt < 3; ++attempt) {
			int result = 0;
			REQUIRE_NOTHROW(result = bpftime::bpftime_import_shm_handler_from_json(
					       fixture.shm, 1, payload.c_str()));
			CHECK(result < 0);
			CHECK(observer.get_free_memory() == constrained_free);
			CHECK(fixture.snapshot() == original);
		}
		break;
	}
	REQUIRE(reached_refcount_failure);
	CHECK(observer.get_free_memory() == free_before);
}

TEST_CASE("JSON preserves export-only epoll and memfd metadata", "[json]")
{
	json_fixture source;
	json_fixture restored;
	// The JSON contract concerns shared metadata, not OS-fd reservation.
	// Use the same handler constructors as the public creation methods,
	// without opening a host device merely to choose an unused handler ID.
	boost::interprocess::managed_shared_memory memory(
		boost::interprocess::open_only, source.name.c_str());
	auto &manager = const_cast<handler_manager &>(*source.shm.get_manager());
	constexpr int allocated_fd = 40;
	json expected;
	SECTION("epoll retains its legacy type-only record")
	{
		REQUIRE(manager.set_handler(allocated_fd, epoll_handler(memory),
					    memory) == allocated_fd);
		expected = { { "type", "epoll_handler" } };
	}
	SECTION("memfd retains its name and flags")
	{
		const char *name = "json_\"metadata\"\nname";
		REQUIRE(manager.set_handler(allocated_fd, memfd_handler(name, 3, memory),
					    memory) == allocated_fd);
		expected = { { "type", "memfd_handler" },
			     { "attr", { { "flags", 3 }, { "name", name } } } };
	}
	constexpr int map_fd = 8;
	constexpr int existing_fd = 9;
	add_map(source.shm, map_fd);
	add_map(restored.shm, existing_fd);
	const auto original = restored.snapshot();
	const auto document = source.snapshot();
	REQUIRE(document.size() == 2);
	CHECK(document.at(std::to_string(allocated_fd)) == expected);
	CHECK(document.at(std::to_string(map_fd)).at("type") == "bpf_map_handler");

	// Decode a valid map before the unsupported record, regardless of the
	// metadata handler ID above. Keys sort numerically and lexically.
	const json input{
		{ "20", document.at(std::to_string(map_fd)) },
		{ "30", document.at(std::to_string(allocated_fd)) }
	};
	REQUIRE(input.begin().key() == "20");
	source.write(input.dump());
	// Neither the old importer nor this transactional importer restores these
	// handlers. Their presence rejects the entire input without publishing the map.
	errno = 0;
	CHECK(bpftime_import_shm_from_json(restored.shm, source.path.c_str()) < 0);
	CHECK(errno == EINVAL);
	CHECK_FALSE(restored.shm.get_manager()->is_allocated(20));
	CHECK_FALSE(restored.shm.get_manager()->is_allocated(30));
	CHECK(restored.snapshot() == original);
	errno = 0;
	CHECK(bpftime::bpftime_import_shm_handler_from_json(
		      restored.shm, allocated_fd, expected.dump().c_str()) < 0);
	CHECK(errno == EINVAL);
	CHECK(restored.snapshot() == original);
}

TEST_CASE("JSON legacy empty snapshots normalize to an object", "[json]")
{
	json_fixture fixture;
	CHECK(fixture.snapshot() == json::object());
	for (const auto *empty : { "null", "{}" }) {
		fixture.write(empty);
		REQUIRE(bpftime_import_shm_from_json(fixture.shm, fixture.path.c_str()) == 0);
		CHECK(fixture.snapshot() == json::object());
	}
}
