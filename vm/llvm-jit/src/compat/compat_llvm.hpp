#ifndef _BPFTIME_VM_COMPAT_LLVM_HPP
#define _BPFTIME_VM_COMPAT_LLVM_HPP
#include <bpftime_vm_compat.hpp>
#include <memory>
#include <vector>
#include <ebpf_inst.h>
class llvm_bpf_jit_context;
namespace bpftime::vm::llvm
{
struct external_function {
	std::string name;
	void *fn;
};

class bpftime_llvm_jit_vm : public bpftime::vm::compat::bpftime_vm_impl {
    public:
	bpftime_llvm_jit_vm();
	std::string get_error_message();
	int register_external_function(size_t index, const std::string &name,
				       void *fn);
	int load_code(const void *code, size_t code_len);
	void unload_code();
	int exec(void *mem, size_t mem_len, uint64_t &bpf_return_value);
	std::optional<compat::precompiled_ebpf_function> compile();
	void set_lddw_helpers(uint64_t (*map_by_fd)(uint32_t),
			      uint64_t (*map_by_idx)(uint32_t),
			      uint64_t (*map_val)(uint64_t),
			      uint64_t (*var_addr)(uint32_t),
			      uint64_t (*code_addr)(uint32_t));

    private:
	uint64_t (*map_by_fd)(uint32_t) = nullptr;
	uint64_t (*map_by_idx)(uint32_t) = nullptr;
	uint64_t (*map_val)(uint64_t) = nullptr;
	uint64_t (*var_addr)(uint32_t) = nullptr;
	uint64_t (*code_addr)(uint32_t) = nullptr;
	std::vector<ebpf_inst> instructions;
	std::vector<std::optional<external_function> > ext_funcs;

	std::unique_ptr<class ::llvm_bpf_jit_context> jit_ctx;
	friend class ::llvm_bpf_jit_context;

	std::string error_msg;
};
} // namespace bpftime::vm::llvm

#endif
