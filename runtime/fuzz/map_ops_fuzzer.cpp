/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 *
 * libFuzzer target for the userspace map backends. The input selects a map
 * shape (type, key/value size, capacity, creation flags) and then a sequence
 * of operations that is dispatched through bpf_map_handler, the layer shared
 * by the BPF helpers (from_syscall=false) and the intercepted bpf() syscall
 * (from_syscall=true). Map attributes are kept inside the ranges the kernel
 * accepts at BPF_MAP_CREATE, so a finding is a backend bug rather than a
 * missing create-time check.
 *
 * The same entry point is replayed with fixed inputs by the runtime unit
 * tests, so the harness keeps building and running without a fuzzing
 * toolchain.
 */
#include "bpftime_config.hpp"
#include "bpftime_handler.hpp"
#include "bpftime_shm.hpp"
#include <algorithm>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <linux/bpf.h>
#include <memory>
#include <string>
#include <unistd.h>
#include <variant>
#include <vector>

using boost::interprocess::managed_shared_memory;
using bpftime::bpf_map_handler;
using bpftime::handler_manager;

namespace
{

constexpr int map_fd = 1;
constexpr size_t max_ops_per_input = 256;
constexpr size_t fuzz_shm_size = 64 << 20;

// Reads the input front to back; bytes past the end read as zero so every
// input, including the empty one, maps to a well-defined operation list.
class input_reader {
    public:
	input_reader(const uint8_t *data, size_t size) : data(data), size(size)
	{
	}
	bool empty() const
	{
		return pos >= size;
	}
	uint8_t byte()
	{
		return pos < size ? data[pos++] : 0;
	}
	void fill(uint8_t *dst, size_t len)
	{
		size_t avail = std::min(len, size - pos);
		if (avail != 0)
			memcpy(dst, data + pos, avail);
		memset(dst + avail, 0, len - avail);
		pos += avail;
	}

    private:
	const uint8_t *data;
	size_t size;
	size_t pos = 0;
};

// linux/bpf.h declares a global enum bpf_map_type, so the bpftime one is
// always qualified.
using map_type = bpftime::bpf_map_type;

enum class key_kind { var_key, u32_key, no_key };

struct map_shape {
	map_type type;
	// Array-style maps index by a u32 key; queue/stack maps have no key.
	key_kind key;
	bool lookup_and_delete;
};

constexpr map_shape map_shapes[] = {
	{ map_type::BPF_MAP_TYPE_HASH, key_kind::var_key, true },
	{ map_type::BPF_MAP_TYPE_ARRAY, key_kind::u32_key, false },
	{ map_type::BPF_MAP_TYPE_PERCPU_HASH, key_kind::var_key, true },
	{ map_type::BPF_MAP_TYPE_PERCPU_ARRAY, key_kind::u32_key, false },
	{ map_type::BPF_MAP_TYPE_LRU_HASH, key_kind::var_key, true },
	{ map_type::BPF_MAP_TYPE_QUEUE, key_kind::no_key, true },
	{ map_type::BPF_MAP_TYPE_STACK, key_kind::no_key, true },
};

constexpr uint64_t map_create_flags[] = { 0, BPF_F_RDONLY, BPF_F_WRONLY };

// One shared-memory segment per process, removed again at exit so repeated
// runs leave no /dev/shm residue.
struct fuzz_env {
	std::string name;
	std::unique_ptr<managed_shared_memory> segment;
	handler_manager *manager = nullptr;

	fuzz_env()
		: name("bpftime_map_ops_fuzz_" + std::to_string(getpid()))
	{
		boost::interprocess::shared_memory_object::remove(name.c_str());
		segment = std::make_unique<managed_shared_memory>(
			boost::interprocess::create_only, name.c_str(),
			fuzz_shm_size);
		manager = segment->construct<handler_manager>(
			boost::interprocess::anonymous_instance)(
			*segment, MIN_MAX_FD_COUNT);
	}
	// Every input releases its map before returning, so only the segment
	// itself is left. Nothing here logs: spdlog may already be torn down
	// when static destructors run.
	~fuzz_env()
	{
		segment.reset();
		boost::interprocess::shared_memory_object::remove(name.c_str());
	}
};

fuzz_env &get_env()
{
	static fuzz_env env;
	return env;
}

void run_keyed_op(const bpf_map_handler &map, const map_shape &shape,
		  uint8_t op, input_reader &in, std::vector<uint8_t> &key,
		  std::vector<uint8_t> &next_key, std::vector<uint8_t> &value,
		  uint32_t value_size)
{
	const bool from_syscall = (op & 0x80) != 0;
	// 0..2 are BPF_ANY/BPF_NOEXIST/BPF_EXIST, 3 must be rejected.
	const uint64_t update_flags = (op >> 4) & 0x3;
	in.fill(key.data(), key.size());
	switch (op & 0x7) {
	case 0:
		in.fill(value.data(), value.size());
		map.map_update_elem(key.data(), value.data(), update_flags,
				    from_syscall);
		break;
	case 1:
		if (const void *found =
			    map.map_lookup_elem(key.data(), from_syscall)) {
			// Touch the returned slot so a sanitizer sees an
			// undersized or stale value.
			memcpy(value.data(), found, value_size);
		}
		break;
	case 2:
		map.map_delete_elem(key.data(), from_syscall);
		break;
	case 3:
		map.bpf_map_get_next_key(key.data(), next_key.data(),
					 from_syscall);
		break;
	case 4:
		map.bpf_map_get_next_key(nullptr, next_key.data(),
					 from_syscall);
		break;
	case 5:
		if (shape.lookup_and_delete)
			map.map_lookup_and_delete_elem(key.data(), value.data(),
						       from_syscall);
		break;
	default:
		break;
	}
}

void run_keyless_op(const bpf_map_handler &map, uint8_t op, input_reader &in,
		    std::vector<uint8_t> &value)
{
	const bool from_syscall = (op & 0x80) != 0;
	const uint64_t push_flags = (op >> 4) & 0x3;
	switch (op & 0x3) {
	case 0:
		in.fill(value.data(), value.size());
		map.map_push_elem(value.data(), push_flags, from_syscall);
		break;
	case 1:
		map.map_pop_elem(value.data(), from_syscall);
		break;
	case 2:
		map.map_peek_elem(value.data(), from_syscall);
		break;
	default:
		map.map_lookup_and_delete_elem(nullptr, value.data(),
					       from_syscall);
		break;
	}
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
	auto &env = get_env();
	input_reader in(data, size);

	const auto &shape =
		map_shapes[in.byte() % (sizeof(map_shapes) / sizeof(map_shapes[0]))];
	uint32_t key_size = 1 + in.byte() % 64;
	const uint32_t value_size = 1 + in.byte() % 256;
	const uint32_t max_entries = 1 + in.byte() % 64;
	const uint64_t create_flags =
		map_create_flags[in.byte() % (sizeof(map_create_flags) /
					      sizeof(map_create_flags[0]))];
	if (shape.key == key_kind::u32_key)
		key_size = sizeof(uint32_t);
	else if (shape.key == key_kind::no_key)
		key_size = 0;

	if (env.manager->set_handler(
		    map_fd,
		    bpf_map_handler(map_fd, (int)shape.type, key_size,
				    value_size, max_entries, create_flags,
				    "fuzz_map", *env.segment),
		    *env.segment) != map_fd) {
		return 0;
	}
	const auto &map = std::get<bpf_map_handler>((*env.manager)[map_fd]);

	// Per-CPU maps exchange all CPU slots on the syscall path, so size the
	// value buffer for the larger of the two layouts.
	std::vector<uint8_t> key(std::max<uint32_t>(key_size, 1));
	std::vector<uint8_t> next_key(key.size());
	std::vector<uint8_t> value(std::max(map.get_userspace_value_size(),
					    value_size));

	for (size_t i = 0; i < max_ops_per_input && !in.empty(); i++) {
		const uint8_t op = in.byte();
		if (shape.key == key_kind::no_key)
			run_keyless_op(map, op, in, value);
		else
			run_keyed_op(map, shape, op, in, key, next_key, value,
				     value_size);
	}

	env.manager->clear_id_at(map_fd, *env.segment);
	return 0;
}
