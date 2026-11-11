// SPDX-License-Identifier: MIT
// This executable tests only ordinary userspace-array JSON transactions.
// GNU ld wrapping delegates to the real verifier registry; no descriptor stub
// or production fault-injection hook is used.
#include <bpftime-verifier.hpp>
#include <bpftime_shm_json.hpp>
#include <catch2/catch_test_macros.hpp>
#include <json.hpp>
#include <linux/bpf.h>
#include <spdlog/spdlog.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <new>
#include <string>
#include <system_error>
#include <unistd.h>

using bpftime::bpftime_shm;
using bpftime::verifier::BpftimeMapDescriptor;
using descriptors = std::map<int, BpftimeMapDescriptor>;
using json = nlohmann::json;

namespace
{
enum class failure { none, snapshot, commit };

struct observation {
	bpftime_shm *shm = nullptr;
	failure fault = failure::none;
	bool file = false;
	unsigned gets = 0, sets = 0, allocation_failures = 0;
	bool getter_saw_publication = false;
	bool setter_saw_publication = false;
};
thread_local observation observed;
thread_local bool fail_next_allocation = false;

// The fault is armed only inside a wrapped real API call and is always disarmed
// before Catch2, logging, fixture cleanup or another allocation can run.
struct allocation_fault {
	explicit allocation_fault(bool enabled)
	{
		fail_next_allocation = enabled;
	}
	~allocation_fault() { fail_next_allocation = false; }
};
} // namespace

void *operator new(std::size_t size)
{
	if (fail_next_allocation) {
		fail_next_allocation = false;
		++observed.allocation_failures;
		throw std::bad_alloc();
	}
	if (auto result = std::malloc(size ? size : 1))
		return result;
	throw std::bad_alloc();
}

void operator delete(void *ptr) noexcept { std::free(ptr); }
void operator delete(void *ptr, std::size_t) noexcept { std::free(ptr); }

// These Itanium/libstdc++ symbols are restricted to the supported toolchain in
// CMake. An ABI/linker mismatch must not silently substitute a fake registry.
descriptors real_get() asm(
	"__real__ZN7bpftime8verifier19get_map_descriptorsEv");
void real_set(const descriptors &) asm(
	"__real__ZN7bpftime8verifier19set_map_descriptorsERKSt3mapIiNS0_20BpftimeMapDescriptorESt4lessIiESaISt4pairIKiS2_EEE");
descriptors wrapped_get() asm(
	"__wrap__ZN7bpftime8verifier19get_map_descriptorsEv");
void wrapped_set(const descriptors &) asm(
	"__wrap__ZN7bpftime8verifier19set_map_descriptorsERKSt3mapIiNS0_20BpftimeMapDescriptorESt4lessIiESaISt4pairIKiS2_EEE");

descriptors wrapped_get()
{
	++observed.gets;
	// Neither imported handler may be published before the snapshot.
	observed.getter_saw_publication = observed.shm &&
		(observed.shm->get_manager()->is_allocated(40) ||
		 (observed.file &&
		  observed.shm->get_manager()->is_allocated(41)));
	allocation_fault fault(observed.fault == failure::snapshot);
	return real_get();
}

void wrapped_set(const descriptors &maps)
{
	++observed.sets;
	observed.setter_saw_publication = observed.shm &&
		observed.shm->get_manager()->is_allocated(40) &&
		(!observed.file ||
		 observed.shm->get_manager()->is_allocated(41));
	allocation_fault fault(observed.fault == failure::commit);
	real_set(maps);
}

namespace
{
struct temporary_directory {
	std::string path;

	temporary_directory()
	{
		const char *root = std::getenv("TMPDIR");
		path = std::string(root && *root ? root : "/tmp") +
			"/bpftime-json-verifier-XXXXXX";
		if (!mkdtemp(path.data()))
			throw std::system_error(errno, std::generic_category(),
						"mkdtemp");
	}
	~temporary_directory()
	{
		std::error_code error;
		std::filesystem::remove_all(path, error);
	}
};

// Ownership is acquired exclusively. A collision throws without unlinking
// somebody else's segment. This member is fully constructed before bpftime_shm
// or any assertion, so fixture-construction exceptions also unlink our segment.
struct owned_segment {
	std::string name;
	boost::interprocess::managed_shared_memory view;

	explicit owned_segment(const std::string &directory)
		: name(std::filesystem::path(directory).filename().string()),
		  view(boost::interprocess::create_only, name.c_str(), 20U << 20)
	{
	}
	~owned_segment()
	{
		boost::interprocess::shared_memory_object::remove(name.c_str());
	}
};

struct observation_scope {
	explicit observation_scope(bpftime_shm &shm)
	{
		observed = {};
		observed.shm = &shm;
	}
	~observation_scope()
	{
		fail_next_allocation = false;
		observed = {};
		// This dedicated process owns its registry; an empty replacement does
		// not allocate and cannot retain descriptors from a failed fixture.
		real_set({});
	}
};

bool same(const BpftimeMapDescriptor &a, const BpftimeMapDescriptor &b)
{
	return a.original_fd == b.original_fd && a.type == b.type &&
	       a.key_size == b.key_size && a.value_size == b.value_size &&
	       a.max_entries == b.max_entries && a.inner_map_fd == b.inner_map_fd;
}

json array_definition()
{
	return { { "type", "bpf_map_handler" },
		 { "name", "ordinary-array" },
		 { "attr",
		   { { "map_type", BPF_MAP_TYPE_ARRAY },
		     { "key_size", 4 },
		     { "value_size", 8 },
		     { "max_entries", 2 } } } };
}

struct fixture {
	temporary_directory directory;
	owned_segment segment{ directory.path };
	bpftime_shm shm{ segment.name.c_str(),
			bpftime::shm_open_type::SHM_CREATE_OR_OPEN };
	observation_scope hooks{ shm };
	const BpftimeMapDescriptor original{ 8, BPF_MAP_TYPE_ARRAY, 4, 8, 1,
					    unsigned(-1) };
	uint32_t key = 0;
	uint64_t value = 0x123456789abcdef0ULL;

	fixture()
	{
		spdlog::set_level(spdlog::level::off);
		REQUIRE(shm.add_bpf_map(8, "existing",
				       { .type = BPF_MAP_TYPE_ARRAY,
					 .key_size = 4,
					 .value_size = 8,
					 .max_ents = 1 }) == 8);
		REQUIRE(shm.bpf_map_update_elem(8, &key, &value, 0, false) == 0);
		real_set({ { 8, original } });
		// Seeding the pre-existing map also uses the descriptor APIs.
		observed = {};
		observed.shm = &shm;
	}

	int import(bool file)
	{
		const auto definition = array_definition();
		if (!file)
			return bpftime::bpftime_import_shm_handler_from_json(
				shm, 40, definition.dump().c_str());
		const auto path = directory.path + "/input.json";
		std::ofstream output(path);
		output << json{ { "40", definition }, { "41", definition } }.dump();
		output.close();
		REQUIRE(output.good());
		return bpftime::bpftime_import_shm_from_json(shm, path.c_str());
	}

	void require_original_preserved()
	{
		REQUIRE(shm.get_manager()->is_allocated(8));
		const auto actual = static_cast<const uint64_t *>(
			shm.bpf_map_lookup_elem(8, &key, false));
		REQUIRE(actual != nullptr);
		REQUIRE(*actual == value);
		const auto maps = real_get();
		REQUIRE(maps.count(8) == 1);
		REQUIRE(same(maps.at(8), original));
	}
};

void check_transaction(bool file, failure fault)
{
	fixture f;
	const auto free_before = f.segment.view.get_free_memory();
	observed.file = file;
	observed.fault = fault;

	const auto result = f.import(file);
	// Disarm before making assertions, including after a regression fails to
	// call a wrapper. allocation_failures proves a real allocation was hit.
	observed.fault = failure::none;
	REQUIRE_FALSE(fail_next_allocation);
	REQUIRE(result == (fault == failure::none ? 0 : -1));
	REQUIRE(observed.gets == 1);
	REQUIRE(observed.sets == (fault == failure::snapshot ? 0U : 1U));
	REQUIRE(observed.allocation_failures ==
		(fault == failure::none ? 0U : 1U));
	REQUIRE_FALSE(observed.getter_saw_publication);
	REQUIRE(observed.setter_saw_publication ==
		(fault != failure::snapshot));

	const auto maps = real_get();
	if (fault == failure::none) {
		REQUIRE(maps.size() == (file ? 3 : 2));
		for (int fd = 40; fd <= (file ? 41 : 40); ++fd) {
			REQUIRE(f.shm.get_manager()->is_allocated(fd));
			REQUIRE(maps.count(fd) == 1);
			REQUIRE(same(maps.at(fd),
				     { fd, BPF_MAP_TYPE_ARRAY, 4, 8, 2,
				       unsigned(-1) }));
		}
	} else {
		REQUIRE(maps.size() == 1);
		REQUIRE_FALSE(f.shm.get_manager()->is_allocated(40));
		REQUIRE_FALSE(f.shm.get_manager()->is_allocated(41));
		REQUIRE(f.segment.view.get_free_memory() == free_before);
	}
	f.require_original_preserved();
}
} // namespace

TEST_CASE("JSON array load commits real verifier descriptors")
{
	check_transaction(false, failure::none);
}

TEST_CASE("JSON array load snapshot failure leaves state unchanged")
{
	check_transaction(false, failure::snapshot);
}

TEST_CASE("JSON array load commit failure rolls back published handlers")
{
	check_transaction(false, failure::commit);
}

TEST_CASE("JSON array file import commits one descriptor batch")
{
	check_transaction(true, failure::none);
}

TEST_CASE("JSON array file snapshot failure leaves state unchanged")
{
	check_transaction(true, failure::snapshot);
}

TEST_CASE("JSON array file commit failure rolls back both handlers")
{
	check_transaction(true, failure::commit);
}
