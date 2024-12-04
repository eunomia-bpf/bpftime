#include "bpf/bpf.h"
#include "bpf/libbpf_common.h"
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include "catch2/catch_test_macros.hpp"
#include "linux/bpf.h"
#include "spdlog/spdlog.h"
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <handler/handler_manager.hpp>
#include <handler/map_handler.hpp>
#include <optional>
#include <thread>
#include <unistd.h>

using boost::interprocess::managed_shared_memory;

const char *SHM_NAME = "bpftime_map_handler_test";

struct testable_map_def {
	bpftime::bpf_map_type map_type;
	std::optional<int> kernel_map_type;
	bool is_per_cpu = false;
	uint64_t extra_flags = 0;
	bool can_delete = true;
	bool create_only = false;
	std::optional<uint32_t> value_size_hack;
};

static testable_map_def testable_maps[] = {
	{
		.map_type = bpftime::bpf_map_type::BPF_MAP_TYPE_HASH,
	},
	{ .map_type = bpftime::bpf_map_type::BPF_MAP_TYPE_ARRAY,
	  .can_delete = false },
	{ .map_type = bpftime::bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY,
	  .is_per_cpu = true,
	  .can_delete = false },
	{ .map_type = bpftime::bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH,
	  .is_per_cpu = true },
	{ .map_type = bpftime::bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY,
	  .kernel_map_type = BPF_MAP_TYPE_ARRAY,
	  .extra_flags = BPF_F_MMAPABLE,
	  .can_delete = false },
	{ .map_type = bpftime::bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_HASH,
	  .kernel_map_type = BPF_MAP_TYPE_HASH },
	{ .map_type = bpftime::bpf_map_type::BPF_MAP_TYPE_RINGBUF,
	  .create_only = true,
	  .value_size_hack = 4 },
	{ .map_type = bpftime::bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY,
	  .create_only = true,
	  .value_size_hack = 4 },

	{ .map_type = bpftime::bpf_map_type::BPF_MAP_TYPE_PROG_ARRAY,
	  .create_only = true,
	  .value_size_hack = 4 },
	{ .map_type = bpftime::bpf_map_type::BPF_MAP_TYPE_ARRAY_OF_MAPS,
	  .create_only = true },

};

extern "C" {
uint64_t bpf_ringbuf_discard(uint64_t data, uint64_t flags, uint64_t, uint64_t,
			     uint64_t);
uint64_t bpf_ringbuf_submit(uint64_t data, uint64_t flags, uint64_t, uint64_t,
			    uint64_t);
uint64_t bpf_ringbuf_reserve(uint64_t rb, uint64_t size, uint64_t flags,
			     uint64_t, uint64_t);
uint64_t bpf_ringbuf_output(uint64_t rb, uint64_t data, uint64_t size,
			    uint64_t flags, uint64_t);
}

TEST_CASE("Test map handler")
{
	struct bpftime::shm_remove remover(SHM_NAME);
	int ncpu = sysconf(_SC_NPROCESSORS_ONLN);

	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
	auto &shm = bpftime::shm_holder.global_shared_memory;

	auto &manager_ref = shm.get_manager_mut();
	auto &segment = shm.get_segment_manager();

	SECTION("Test good operations")
	{
		for (auto map_type : testable_maps) {
			SPDLOG_INFO("Testing map type {}",
				    (int)map_type.map_type);
			const uint32_t expected_value_size =
				map_type.value_size_hack.value_or(8);
			struct kernel_map_tuple {
				uint32_t id;
				int fd;
			};
			std::optional<kernel_map_tuple> kernel_map_info;
			if (map_type.kernel_map_type) {
				LIBBPF_OPTS(bpf_map_create_opts, opts);
				opts.map_flags = map_type.extra_flags;
				int fd = bpf_map_create(
					(enum bpf_map_type)map_type
						.kernel_map_type.value(),
					"test_map", 4, expected_value_size,
					1024, &opts);
				REQUIRE(fd > 0);
				struct bpf_map_info info;
				uint32_t len = sizeof(info);
				REQUIRE(bpf_map_get_info_by_fd(fd, &info,
							       &len) == 0);
				kernel_map_info = {
					.id = info.id,
					.fd = fd,
				};
				SPDLOG_INFO("Created kernel map, fd={}, id={}",
					    fd, info.id);
			}
			manager_ref.set_handler(
				1,
				bpftime::bpf_map_handler(
					1, "test_map", segment,
					bpftime::bpf_map_attr{
						.type = (int)map_type.map_type,
						.key_size = 4,
						.value_size =
							expected_value_size,
						.max_ents = 1024,
						.kernel_bpf_map_id =
							kernel_map_info ?
								kernel_map_info
									->id :
								0 }),
				segment);

			auto &map = std::get<bpftime::bpf_map_handler>(
				manager_ref[1]);

			if (!map_type.create_only) {
				if (map_type.is_per_cpu) {
					REQUIRE(map.get_value_size() ==
						(uint32_t)expected_value_size *
							ncpu);
				} else {
					REQUIRE(map.get_value_size() ==
						expected_value_size);
				}
				int32_t key = 0;
				uint64_t value = 666;
				REQUIRE(map.map_update_elem(&key, &value, 0,
							    false) == 0);
				key = 1;
				REQUIRE(map.map_update_elem(&key, &value, 0,
							    false) == 0);
				int32_t next_key = 0;
				int32_t test_key = 21000;

				REQUIRE(map.bpf_map_get_next_key(
						nullptr, &next_key) == 0);
				if (map_type.is_per_cpu) {
					auto valueptr =
						(uint64_t *)map.map_lookup_elem(
							&key, true);
					REQUIRE(valueptr != nullptr);
					bool found = false;
					for (int i = 0; i < ncpu; i++) {
						if (valueptr[i] == value) {
							found = true;
							break;
						}
					}
					REQUIRE(found);
				} else {
					auto valueptr = map.map_lookup_elem(
						&key, false);
					REQUIRE(valueptr != nullptr);
					REQUIRE(*(uint64_t *)valueptr == value);
				}
				if (map_type.can_delete) {
					REQUIRE(map.map_delete_elem(&key) == 0);
					if (!map_type.is_per_cpu) {
						auto valueptr =
							map.map_lookup_elem(
								&key, false);
						REQUIRE(valueptr == nullptr);
					}
				}
			}
			manager_ref.clear_id_at(1, segment);
			if (kernel_map_info) {
				close(kernel_map_info->id);
			}
		}
	}
	SECTION("Test bad map types")
	{
		manager_ref.set_handler(
			1,
			bpftime::bpf_map_handler(
				1, "test_map", segment,
				bpftime::bpf_map_attr{ .type = (int)-1,
						       .key_size = 4,
						       .value_size = 8,
						       .max_ents = 1024 }),
			segment);
		auto &map = std::get<bpftime::bpf_map_handler>(manager_ref[1]);

		REQUIRE(map.map_update_elem(nullptr, nullptr, 0) < 0);
		REQUIRE(map.map_lookup_elem(nullptr) == nullptr);
		REQUIRE(map.map_delete_elem(nullptr) < 0);
		REQUIRE(map.bpf_map_get_next_key(nullptr, nullptr) < 0);
		manager_ref.clear_id_at(1, segment);
	}
	SECTION("Test ringbuf")
	{
		shm.add_bpf_map(1, "test_map",

				bpftime::bpf_map_attr{
					.type = (int)bpftime::bpf_map_type::
						BPF_MAP_TYPE_RINGBUF,
					.key_size = 4,
					.value_size = 8,
					.max_ents = 1 << 20 });
		auto &map = std::get<bpftime::bpf_map_handler>(manager_ref[1]);
		REQUIRE((map.map_lookup_elem(nullptr) == nullptr &&
			 errno == ENOTSUP));
		REQUIRE((map.map_update_elem(nullptr, nullptr, 0) < 0 &&
			 errno == ENOTSUP));
		REQUIRE((map.map_delete_elem(nullptr) < 0 && errno == ENOTSUP));
		REQUIRE((map.bpf_map_get_next_key(nullptr, nullptr) < 0 &&
			 errno == ENOTSUP));
		{
			auto rb_ptr = ((uint64_t)1 << 32);
			std::vector<std::thread> thds;
			SECTION("Test with reserve+commit")
			{
				thds.push_back(std::thread([=]() {
					for (int i = 1; i <= 100; i++) {
						void *ptr = (void *)(uintptr_t)
							bpf_ringbuf_reserve(
								rb_ptr,
								sizeof(int), 0,
								0, 0);
						REQUIRE(ptr != nullptr);
						memcpy(ptr, &i, sizeof(int));
						if (i % 2 == 0) {
							// discard it
							bpf_ringbuf_discard(
								(uintptr_t)ptr,
								0, 0, 0, 0);
						} else {
							// submit it
							bpf_ringbuf_submit(
								(uintptr_t)ptr,
								0, 0, 0, 0);
						}
					}
				}));
			}
			SECTION("Test with output")
			{
				thds.push_back(std::thread([=]() {
					for (int i = 1; i <= 100; i += 2) {
						bpf_ringbuf_output(
							rb_ptr, (uintptr_t)&i,
							sizeof(i), 0, 0);
					}
				}));
			}
			auto impl = map.try_get_ringbuf_map_impl()
					    .value()
					    ->create_impl_shared_ptr();
			thds.push_back(std::thread([=]() {
				std::vector<int> data;
				while (data.size() < 50) {
					impl->fetch_data([&](void *buf,
							     int len) {
						REQUIRE(len == sizeof(int));
						data.push_back(*(int *)buf);
						return 0;
					});
				}
				for (int i = 0; i < (int)data.size(); i++)
					REQUIRE(data[i] == 2 * i + 1);
			}));
			for (auto &thd : thds)
				thd.join();
		}
		auto weak_ptr = map.try_get_ringbuf_map_impl()
					.value()
					->create_impl_weak_ptr();

		manager_ref.clear_id_at(1, segment);
		REQUIRE(weak_ptr.expired());
	}
	bpftime_destroy_global_shm();
}
