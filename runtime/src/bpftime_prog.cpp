/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime.hpp"
#include "bpftime_helper_group.hpp"
#include "bpftime_internal.h"
#include "ebpf-vm.h"
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <spdlog/spdlog.h>

using namespace std;
namespace bpftime
{

thread_local std::optional<uint64_t> current_thread_bpf_cookie;

bpftime_prog::bpftime_prog(const struct ebpf_inst *insn, size_t insn_cnt,
			   const char *name)
	: name(name)
{
	SPDLOG_DEBUG("Creating bpftime_prog with name {}", name);
	insns.assign(insn, insn + insn_cnt);
	vm = ebpf_create();
	// Disable bounds check because we have no implementation yet
	// ebpf_toggle_bounds_check(vm, false);
	ebpf_set_lddw_helpers(vm, map_ptr_by_fd, nullptr, map_val, nullptr,
			      nullptr);
}

bpftime_prog::~bpftime_prog()
{
	ebpf_unload_code(vm);
	ebpf_destroy(vm);
}

int bpftime_prog::bpftime_prog_load(bool jit)
{
	int res = -1;

	SPDLOG_DEBUG("Load insn cnt {}", insns.size());
	res = ebpf_load(vm, insns.data(),
			insns.size() * sizeof(struct ebpf_inst), &errmsg);
	if (res < 0) {
		SPDLOG_ERROR("Failed to load insn: {}", errmsg);
		return res;
	}
	if (jit) {
		// run with jit mode
		jitted = true;
		ebpf_jit_fn jit_fn = ebpf_compile(vm, &errmsg);
		if (jit_fn == NULL) {
			SPDLOG_ERROR("Failed to compile: {}", errmsg);
			return -1;
		}
		fn = jit_fn;
	} else {
		// ignore for vm
		jitted = false;
	}
	return 0;
}

int bpftime_prog::bpftime_prog_unload()
{
	if (jitted) {
		// ignore for jit
		return 0;
	}
	ebpf_unload_code(vm);
	return 0;
}

int bpftime_prog::bpftime_prog_exec(void *memory, size_t memory_size,
				    uint64_t *return_val) const
{
	uint64_t val = 0;
	int res = 0;
	// set share memory read and write able
	bpftime_protect_disable();
	SPDLOG_DEBUG(
		"Calling bpftime_prog::bpftime_prog_exec, memory={:x}, memory_size={}, return_val={:x}, prog_name={}",
		(uintptr_t)memory, memory_size, (uintptr_t)return_val,
		this->name);
	if (jitted) {
		SPDLOG_DEBUG("Directly call jitted function at {:x}",
			     (uintptr_t)fn);
		val = fn(memory, memory_size);
	} else {
		SPDLOG_DEBUG("Running using ebpf_exec");
		res = ebpf_exec(vm, memory, memory_size, &val);
		if (res < 0) {
			SPDLOG_ERROR("ebpf_exec returned error: {}", res);
		}
	}
	*return_val = val;
	// set share memory read only
	bpftime_protect_enable();
	return res;
}

int bpftime_prog::bpftime_prog_register_raw_helper(
	struct bpftime_helper_info info)
{
	return ebpf_register(vm, info.index, info.name.c_str(), info.fn);
}

int bpftime_prog::load_aot_object(const std::vector<uint8_t> &buf)
{
	ebpf_jit_fn res = ebpf_load_aot_object(vm, buf.data(), buf.size());
	if (res == nullptr) {
		SPDLOG_ERROR("Failed to load aot object");
		return -1;
	}
	fn = res;
	jitted = true;
	return 0;
}

} // namespace bpftime
