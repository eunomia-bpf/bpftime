#include "spdlog/spdlog.h"
#include <bpftime_vm_compat.hpp>
#include <cerrno>
#include <cstdint>
#include <memory>
#include <ebpf_inst.h>
#include "compat_llvm.hpp"
#include "../llvm/llvm_jit_context.hpp"

namespace bpftime::vm::compat
{

std::unique_ptr<bpftime_vm_impl> create_vm_instance()
{
	return std::make_unique<llvm::bpftime_llvm_jit_vm>();
}

} // namespace bpftime::vm::compat

using namespace bpftime::vm::llvm;

bpftime_llvm_jit_vm::bpftime_llvm_jit_vm() : ext_funcs(MAX_EXT_FUNCS)

{
	this->jit_ctx = std::make_unique<llvm_bpf_jit_context>(this);
}

std::string bpftime_llvm_jit_vm::get_error_message()
{
	return error_msg;
}

int bpftime_llvm_jit_vm::register_external_function(size_t index,
						    const std::string &name,
						    void *fn)
{
	if (index >= ext_funcs.size()) {
		error_msg = "Index too large";
		return -E2BIG;
	}
	if (ext_funcs[index].has_value()) {
		error_msg = "Already defined";
		return -EEXIST;
	}
	ext_funcs[index] = external_function{ .name = name, .fn = fn };
	return 0;
}

int bpftime_llvm_jit_vm::load_code(const void *code, size_t code_len)
{
	if (code_len % 8 != 0) {
		error_msg = "Code len must be a multiple of 8";
		return -EINVAL;
	}
	instructions.assign((ebpf_inst *)code,
			    (ebpf_inst *)code + code_len / 8);
	return 0;
}

void bpftime_llvm_jit_vm::unload_code()
{
	instructions.clear();
}

int bpftime_llvm_jit_vm::exec(void *mem, size_t mem_len,
			      uint64_t &bpf_return_value)
{
	if (jitted_function) {
		SPDLOG_DEBUG("llvm-jit: Called jitted function {:x}",
			     (uintptr_t)jitted_function.value());
		auto ret =
			(*jitted_function)(mem, static_cast<uint64_t>(mem_len));
		SPDLOG_DEBUG(
			"LLJIT: called from jitted function {:x} returned {}",
			(uintptr_t)jitted_function.value(), ret);
		bpf_return_value = ret;
		return 0;
	}
	auto func = compile();
	if (!func) {
		SPDLOG_ERROR("Unable to compile eBPF program");
		return -1;
	}
	SPDLOG_DEBUG("LLJIT: compiled function: {:x}", (uintptr_t)func.value());
	// after compile, run
	return exec(mem, mem_len, bpf_return_value);
}

std::optional<bpftime::vm::compat::precompiled_ebpf_function>
bpftime_llvm_jit_vm::compile()
{
	auto func = jit_ctx->compile();
	jitted_function = func;
	return func;
}

void bpftime_llvm_jit_vm::set_lddw_helpers(uint64_t (*map_by_fd)(uint32_t),
					   uint64_t (*map_by_idx)(uint32_t),
					   uint64_t (*map_val)(uint64_t),
					   uint64_t (*var_addr)(uint32_t),
					   uint64_t (*code_addr)(uint32_t))
{
	this->map_by_fd = map_by_fd;
	this->map_by_idx = map_by_idx;
	this->map_val = map_val;
	this->var_addr = var_addr;
	this->code_addr = code_addr;
}

std::vector<uint8_t> bpftime_llvm_jit_vm::do_aot_compile(bool print_ir)
{
	return this->jit_ctx->do_aot_compile(print_ir);
}

std::optional<bpftime::vm::compat::precompiled_ebpf_function>
bpftime_llvm_jit_vm::load_aot_object(const std::vector<uint8_t> &object)
{
	if (jitted_function) {
		error_msg = "Already compiled";
		return {};
	}
	this->jit_ctx->load_aot_object(object);
	jitted_function = this->jit_ctx->get_entry_address();
	return jitted_function;
}
