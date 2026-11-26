#include <bpftime_shm_json.hpp>
#include <bpftime_shm_internal.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_message.hpp>
#include <json.hpp>
#include <linux/bpf.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>
#include "common_def.hpp"

using namespace bpftime;
using json = nlohmann::json;

namespace
{
struct json_fixture {
	static unsigned next_id;
	std::string name = "bpftime-json-" + std::to_string(getpid()) + "-" +
			   std::to_string(++next_id);
	::shm_remove remover{ std::string(name) };
	bpftime_shm shm{ name.c_str(), shm_open_type::SHM_REMOVE_AND_CREATE };
	shm_write_window_guard write_window{ shm };
	std::string directory;
	std::string path;

	json_fixture()
	{
		auto pattern = (std::filesystem::temp_directory_path() /
				"bpftime-json-XXXXXX").string();
		auto result = mkdtemp(pattern.data());
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

TEST_CASE("JSON instruction hex decoding is strict", "[json]")
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

TEST_CASE("JSON definition graph round-trips with existing export", "[json]")
{
	json_fixture source;
	json_fixture restored;
	add_program(source.shm, 20);
	add_map(source.shm, 8);
	REQUIRE(source.shm.add_bpf_map(9, "json_percpu_array",
				     { .type = BPF_MAP_TYPE_PERCPU_ARRAY, .key_size = 4,
				       .value_size = 4, .max_ents = 10 }) == 9);
	REQUIRE(source.shm.add_tracepoint(30, -1, 1) == 30);
	// The link is read before the endpoints in fd order.
	bpf_link_create_args args{ .prog_fd = 20, .target_fd = 30 };
	REQUIRE(source.shm.add_bpf_link(10, &args) == 10);
	REQUIRE(source.shm.add_software_perf_event(40, 0, 1024, 10) == 40);
	REQUIRE(source.shm.perf_event_enable(40) == 0);
	const auto first = source.snapshot();
	REQUIRE(bpftime_import_shm_from_json(restored.shm, source.path.c_str()) == 0);
	CHECK(restored.snapshot() == first);
	const auto *manager = restored.shm.get_manager();
	const auto &link = std::get<bpf_link_handler>(manager->get_handler(10));
	CHECK(link.prog_id == 20);
	CHECK(link.attach_target_id == 30);
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
	for (int invalid_fd : { -1, STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO,
				static_cast<int>(fixture.shm.get_manager()->size()) }) {
		int result = 0;
		REQUIRE_NOTHROW(result = bpftime::bpftime_import_shm_handler_from_json(
				       fixture.shm, invalid_fd, original["8"].dump().c_str()));
		CHECK(result < 0);
		CHECK(fixture.snapshot() == original);
	}
	CHECK(bpftime::bpftime_import_shm_handler_from_json(fixture.shm, 100, nullptr) < 0);
	CHECK(bpftime_import_shm_from_json(fixture.shm, nullptr) < 0);
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
	json invalid = { { "40", original["8"] },
			 { "100", { { "type", "unknown_handler" } } } };
	fixture.write(invalid.dump());
	int result = 0;
	REQUIRE_NOTHROW(result = bpftime_import_shm_from_json(fixture.shm, fixture.path.c_str()));
	CHECK(result < 0);
	CHECK(fixture.snapshot() == original);
	// Definitions are checked before an otherwise valid map can be installed.
	auto external_map = original["8"];
	external_map["attr"]["map_type"] =
		static_cast<int>(bpftime::bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY);
	auto oversized_map = original["8"];
	oversized_map["attr"]["max_entries"] = UINT32_MAX;
	const std::vector<json> invalid_definitions = {
		external_map,
		oversized_map,
		{ { "type", "bpf_link_handler" },
		  { "attr", { { "prog_fd", 8 }, { "target_fd", 30 } } } },
		{ { "type", "bpf_link_handler" },
		  { "attr", { { "prog_fd", 20 }, { "target_fd", 101 } } } },
		{ { "type", "epoll_handler" } },
	};
	for (const auto &definition : invalid_definitions) {
		fixture.write(json{ { "40", original["8"] }, { "100", definition } }.dump());
		CHECK(bpftime_import_shm_from_json(fixture.shm, fixture.path.c_str()) < 0);
		CHECK(fixture.snapshot() == original);
	}
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
	const json input = { { "40", original["8"] }, { "41", oversized_map } };
	for (int attempt = 0; attempt < 3; ++attempt) {
		fixture.write(input.dump());
		int result = 0;
		REQUIRE_NOTHROW(result = bpftime_import_shm_from_json(
				       fixture.shm, fixture.path.c_str()));
		const int error = errno;
		CHECK(result == -1);
		CHECK(error == ENOMEM);
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
			bpf_map_handler probe(40, "json_map", observer, attr);
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
					       fixture.shm, 40, payload.c_str()));
			const int error = errno;
			CHECK(result == -1);
			CHECK(error == ENOMEM);
			CHECK(observer.get_free_memory() == constrained_free);
			CHECK(fixture.snapshot() == original);
		}
		break;
	}
	REQUIRE(reached_refcount_failure);
	CHECK(observer.get_free_memory() == free_before);
}
