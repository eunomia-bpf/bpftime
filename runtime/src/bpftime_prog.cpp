#include "bpftime.hpp"
#include "bpftime_helper_group.hpp"
#include "bpftime_internal.h"
#include "ebpf-vm.h"
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <spdlog/spdlog.h>

using namespace std;
namespace bpftime
{

bpftime_prog::bpftime_prog(const struct ebpf_inst *insn, size_t insn_cnt,
			   const char *name)
	: name(name)
{
	spdlog::debug("Creating bpftime_prog with name {}", name);
	insns.assign(insn, insn + insn_cnt);
	vm = ebpf_create();
	ebpf_toggle_bounds_check(vm, false);
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

	spdlog::debug("Load insn cnt {}", insns.size());
	res = ebpf_load(vm, insns.data(),
			insns.size() * sizeof(struct ebpf_inst), &errmsg);
	if (res < 0) {
		spdlog::error("Failed to load insn: {}", errmsg);
		return res;
	}
	if (jit) {
		// run with jit mode
		jitted = true;
		ebpf_jit_fn jit_fn = ebpf_compile(vm, &errmsg);
		if (jit_fn == NULL) {
			spdlog::error("Failed to compile: {}", errmsg);
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
	spdlog::debug(
		"Calling bpftime_prog::bpftime_prog_exec, memory={:x}, memory_size={}, return_val={:x}, prog_name={}",
		(uintptr_t)memory, memory_size, (uintptr_t)return_val,
		this->name);
	if (jitted) {
		spdlog::debug("Directly call jitted function at {:x}",
			      (uintptr_t)fn);
		val = fn(memory, memory_size);
	} else {
		spdlog::debug("Running using ebpf_exec");
		res = ebpf_exec(vm, memory, memory_size, &val);
		if (res < 0) {
			spdlog::error("ebpf_exec returned error: {}", res);
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

} // namespace bpftime
