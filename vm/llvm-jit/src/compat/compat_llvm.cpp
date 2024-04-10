#include <bpftime_vm_compat.hpp>
#include <cerrno>
#include <memory>
#include <ebpf_inst.h>
#include <ebpf-vm.h>
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
bpftime_llvm_jit_vm::bpftime_llvm_jit_vm() : instructions(MAX_EXT_FUNCS)

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
	if (ext_funcs[index].has_value()) {
		error_msg = "Already defined";
		return -EEXIST;
	}
	if (index >= ext_funcs.size()) {
		error_msg = "Index too large";
		return -E2BIG;
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
}
std::optional<bpftime::vm::compat::precompiled_ebpf_function>
bpftime_llvm_jit_vm::compile()
{
}
void bpftime_llvm_jit_vm::set_lddw_helpers(uint64_t (*map_by_fd)(uint32_t),
					   uint64_t (*map_by_idx)(uint32_t),
					   uint64_t (*map_val)(uint64_t),
					   uint64_t (*var_addr)(uint32_t),
					   uint64_t (*code_addr)(uint32_t))
{
}
