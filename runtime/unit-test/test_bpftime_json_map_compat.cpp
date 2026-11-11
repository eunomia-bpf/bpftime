// SPDX-License-Identifier: MIT
// Focused JSON definition compatibility tests. No programs, kernel-map access,
// CUDA registration/mapping, or external map lifecycle callbacks are executed.
#include <bpftime_shm_json.hpp>
#include <bpftime_shm_internal.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_message.hpp>
#include <json.hpp>
#include <linux/bpf.h>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>
#include <unistd.h>

using namespace bpftime;
using json = nlohmann::json;

namespace
{
// The broad runtime suite retains its original fixture capacity; the dedicated
// target supplies a smaller bound together with max_fd_count=128 through CTest.
#ifndef BPFTIME_JSON_MAP_TEST_SEGMENT_MB
#define BPFTIME_JSON_MAP_TEST_SEGMENT_MB 20
#endif

struct temporary_directory {
	std::string path;
	temporary_directory()
	{
		const char *root = std::getenv("TMPDIR");
		path = std::string(root && *root ? root : "/tmp") +
			"/bpftime-json-map-XXXXXX";
		if (!mkdtemp(path.data()))
			throw std::system_error(errno, std::generic_category(), "mkdtemp");
	}
	~temporary_directory()
	{
		std::error_code error;
		std::filesystem::remove_all(path, error);
	}
};

// A name collision throws without removing anyone else's shared memory.
// Complete this owner before constructing bpftime_shm so exceptions during
// bpftime_shm initialization also clean up only the segment acquired here.
// If Boost itself fails before construction completes, never unlink blindly:
// a failed create_only call does not establish ownership of a collided name.
struct owned_segment {
	std::string name;
	boost::interprocess::managed_shared_memory view;
	explicit owned_segment(const std::string &directory)
		: name(std::filesystem::path(directory).filename().string()),
		  view(boost::interprocess::create_only, name.c_str(),
		       BPFTIME_JSON_MAP_TEST_SEGMENT_MB * (1U << 20))
	{
	}
	~owned_segment()
	{
		boost::interprocess::shared_memory_object::remove(name.c_str());
	}
};

struct json_fixture {
	temporary_directory directory;
	owned_segment segment{ directory.path };
	bpftime_shm shm{ segment.name.c_str(), shm_open_type::SHM_CREATE_OR_OPEN };
	std::string path = directory.path + "/state.json";

	void write(const std::string &text)
	{
		std::ofstream file(path);
		file << text;
		file.close();
		REQUIRE(file.good());
	}
	json snapshot()
	{
		REQUIRE(bpftime_export_shm_to_json(shm, path.c_str()) == 0);
		std::ifstream file(path);
		REQUIRE(file.is_open());
		return json::parse(file);
	}
};

void add_map(bpftime_shm &shm, int fd)
{
	REQUIRE(shm.add_bpf_map(fd, "json_map",
			       { .type = BPF_MAP_TYPE_ARRAY, .key_size = 4,
				 .value_size = 8, .max_ents = 1 }) == fd);
}

// Serialize from a different segment, then release it before constructing the
// destination. This keeps cross-segment coverage with only one live segment.
json array_record(uint64_t value = 0)
{
	json_fixture source;
	add_map(source.shm, 8);
	const uint32_t key = 0;
	REQUIRE(source.shm.bpf_map_update_elem(8, &key, &value, 0, false) == 0);
	return source.snapshot()["8"];
}

struct map_ops_guard {
	int type;
	bpftime_map_ops previous;
	explicit map_ops_guard(int type)
		: type(type), previous(global_map_ops_table[type]) {}
	~map_ops_guard() { global_map_ops_table[type] = previous; }
};
} // namespace

TEST_CASE("JSON map definitions round-trip and roll back without libbpf",
	  "[json][json-map-compat][json-map-policy]")
{
	const auto record = array_record(42);
	json_fixture restored;
	const uint32_t key = 0;

	SECTION("single-handler load")
	{
		REQUIRE(bpftime::bpftime_import_shm_handler_from_json(
				restored.shm, 8, record.dump().c_str()) == 0);
	}
	SECTION("whole-document import")
	{
		restored.write(json{ { "8", record } }.dump());
		REQUIRE(bpftime_import_shm_from_json(
				restored.shm, restored.path.c_str()) == 0);
	}
	CHECK(restored.snapshot()["8"] == record);
	const auto *empty = static_cast<const uint64_t *>(
		restored.shm.bpf_map_lookup_elem(8, &key, false));
	REQUIRE(empty != nullptr);
	CHECK(*empty == 0); // Definitions do not checkpoint source map contents.

	const auto original = restored.snapshot();
	const auto free_before = restored.segment.view.get_free_memory();
	auto oversized = record;
	oversized["attr"]["max_entries"] =
		restored.segment.view.get_size() / 8 + 1;
	restored.write(json{ { "40", record }, { "41", oversized } }.dump());
	errno = 0;
	CHECK(bpftime_import_shm_from_json(restored.shm, restored.path.c_str()) == -1);
	CHECK(errno == ENOMEM);
	CHECK_FALSE(restored.shm.get_manager()->is_allocated(40));
	CHECK_FALSE(restored.shm.get_manager()->is_allocated(41));
	CHECK(restored.segment.view.get_free_memory() == free_before);
	CHECK(restored.snapshot() == original);
}

TEST_CASE("JSON reserved native maps cannot fall back to custom callbacks",
	  "[json][json-map-compat][json-map-policy]")
{
	std::vector<bpftime::bpf_map_type> rejected{
		bpftime::bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERF_EVENT_ARRAY,
		bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_HASH_MAP,
		bpftime::bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP,
		bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_MAP,
		bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP,
		bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_RINGBUF_MAP,
#ifndef BPFTIME_BUILD_WITH_LIBBPF
		bpftime::bpf_map_type::BPF_MAP_TYPE_PROG_ARRAY,
		bpftime::bpf_map_type::BPF_MAP_TYPE_STACK_TRACE,
		bpftime::bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_HASH,
		bpftime::bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY,
		bpftime::bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERCPU_ARRAY,
#endif
#ifndef BPFTIME_ENABLE_CUDA_ATTACH
		bpftime::bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP,
		bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP,
#endif
	};
	static thread_local unsigned callback_calls;
	callback_calls = 0;
	bpftime_map_ops ops{};
	ops.elem_lookup = +[](int, const void *, bool) -> void * {
		++callback_calls;
		return nullptr;
	};
	json_fixture fixture;
	auto config = fixture.shm.get_runtime_config();
	config.allow_non_buildin_map_types = true;
	fixture.shm.set_runtime_config(std::move(config));
	add_map(fixture.shm, 8);
	const auto original = fixture.snapshot();
	for (const auto type : rejected) {
		const int id = static_cast<int>(type);
		CAPTURE(id);
		map_ops_guard callbacks(id);
		REQUIRE(bpftime_register_map_ops(id, &ops) == 0);
		auto record = original["8"];
		record["attr"]["map_type"] = id;
		for (const bool from_file : { false, true }) {
			CAPTURE(from_file);
			if (from_file)
				fixture.write(json{ { "40", original["8"] },
						    { "41", record } }.dump());
			errno = 0;
			const int result = from_file ?
				bpftime_import_shm_from_json(fixture.shm, fixture.path.c_str()) :
				bpftime::bpftime_import_shm_handler_from_json(
					fixture.shm, 41, record.dump().c_str());
			CHECK(result == -1);
			CHECK(errno == EINVAL);
			CHECK_FALSE(fixture.shm.get_manager()->is_allocated(40));
			CHECK_FALSE(fixture.shm.get_manager()->is_allocated(41));
			CHECK(fixture.snapshot() == original);
			CHECK(callback_calls == 0);
		}
	}
}

#ifdef BPFTIME_ENABLE_CUDA_ATTACH
TEST_CASE("JSON host GPU arrays preserve host storage and metadata", "[json][json-map-compat]")
{
	for (const auto type : {
		     bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP,
		     bpftime::bpf_map_type::BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP }) {
		CAPTURE(static_cast<int>(type));
		const auto ordinary = array_record();
		json_fixture restored;
		auto record = ordinary;
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
			boost::interprocess::open_only, restored.segment.name.c_str());
		const auto free_before = memory.get_free_memory();
		const auto original = restored.snapshot();
		auto oversized = ordinary;
		oversized["attr"]["max_entries"] = memory.get_size() / 8 + 1;
		restored.write(json{ { "1", record }, { "2", oversized } }.dump());
		errno = 0;
		CHECK(bpftime_import_shm_from_json(restored.shm, restored.path.c_str()) < 0);
		CHECK(errno == ENOMEM);
		CHECK(memory.get_free_memory() == free_before);
		CHECK(restored.snapshot() == original);
	}
}

TEST_CASE("JSON host GPU arrays reject inconsistent dimensions", "[json][json-map-compat]")
{
	auto record = array_record();
	json_fixture restored;
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
