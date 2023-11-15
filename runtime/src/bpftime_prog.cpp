#include "bpftime.hpp"
#include "bpftime_helper_group.hpp"
#include "bpftime_internal.h"
#include "ebpf-vm.h"
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <spdlog/spdlog.h>

extern "C" {
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <linux/filter.h>
}

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

int bpftime_prog::bpftime_prog_load(bool jit_enabled)
{
	if (jit_enabled) {
		return bpftime_prog_load(bpftime_vm_type::JIT);
	} else {
		return bpftime_prog_load(bpftime_vm_type::KERNEL);
	}
}

int bpftime_prog::bpftime_prog_load(bpftime_vm_type type)
{
	int res = -1;
	vm_type = type;
	spdlog::debug("Load insn cnt {}", insns.size());
	switch (type) {
	case bpftime_vm_type::KERNEL:
		res = bpf_prog_load(BPF_PROG_TYPE_SOCKET_FILTER, name.c_str(), "GPL",
				    (const bpf_insn *)insns.data(), insns.size(), NULL);
		if (res < 0) {
			spdlog::error("Failed to load insn: {}",
				      strerror(errno));
			return res;
		}
		spdlog::debug("load kernel prog fd: {}", res);
		prog_fd = res;
		break;
	case bpftime_vm_type::VM:
	case bpftime_vm_type::JIT:
		res = ebpf_load(vm, insns.data(),
				insns.size() * sizeof(struct ebpf_inst),
				&errmsg);
		if (res < 0) {
			spdlog::error("Failed to load insn: {}", errmsg);
			return res;
		}
		if (type == bpftime_vm_type::JIT) {
			// run with jit mode
			ebpf_jit_fn jit_fn = ebpf_compile(vm, &errmsg);
			if (jit_fn == NULL) {
				spdlog::error("Failed to compile: {}", errmsg);
				return -1;
			}
			fn = jit_fn;
		}
		break;
	}
	return 0;
}

int bpftime_prog::bpftime_prog_unload()
{
	if (vm_type == bpftime_vm_type::KERNEL ||
	    vm_type == bpftime_vm_type::JIT) {
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
	spdlog::debug(
		"Calling bpftime_prog::bpftime_prog_exec, memory={:x}, memory_size={}, return_val={:x}, prog_name={}",
		(uintptr_t)memory, memory_size, (uintptr_t)return_val,
		this->name);
	switch (vm_type) {
	case bpftime_vm_type::KERNEL: {
		LIBBPF_OPTS(bpf_test_run_opts, opts);
		opts.data_in = memory;
		opts.data_size_in = memory_size;
		spdlog::debug("Running using kernel prog: {}", prog_fd);
		res = bpf_prog_test_run_opts(prog_fd, &opts);
		if (res < 0) {
			spdlog::error("bpf_prog_test_run failed: {}", res);
			return res;
		}
		break;
	}
	case bpftime_vm_type::VM: {
		spdlog::debug("Running using ebpf_exec");
		res = ebpf_exec(vm, memory, memory_size, &val);
		if (res < 0) {
			spdlog::error("ebpf_exec returned error: {}", res);
		}
		break;
	}
	case bpftime_vm_type::JIT: {
		spdlog::debug("Directly call jitted function at {:x}",
			      (uintptr_t)fn);
		val = fn(memory, memory_size);
		break;
	}
	}
	*return_val = val;
	return res;
}

int bpftime_prog::bpftime_prog_register_raw_helper(
	struct bpftime_helper_info info)
{
	return ebpf_register(vm, info.index, info.name.c_str(), info.fn);
}

} // namespace bpftime
