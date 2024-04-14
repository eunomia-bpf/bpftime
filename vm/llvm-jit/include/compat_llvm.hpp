#ifndef _BPFTIME_VM_COMPAT_LLVM_HPP
#define _BPFTIME_VM_COMPAT_LLVM_HPP
#include <bpftime_vm_compat.hpp>
#include <memory>
#include <optional>
#include <vector>
#include <ebpf_inst.h>
#include <llvm/llvm_jit_context.hpp>

namespace bpftime::vm::llvm
{
struct external_function {
	std::string name;
	void *fn;
};

class bpftime_llvm_jit_vm : public bpftime::vm::compat::bpftime_vm_impl {
    public:
	bpftime_llvm_jit_vm();
	/* override */
	std::string get_error_message() override;
	int register_external_function(size_t index, const std::string &name,
				       void *fn) override;
	int load_code(const void *code, size_t code_len) override;
	void unload_code() override;
	int exec(void *mem, size_t mem_len,
		 uint64_t &bpf_return_value) override;
	std::vector<uint8_t> do_aot_compile(bool print_ir = false) override;
	std::optional<compat::precompiled_ebpf_function>
	load_aot_object(const std::vector<uint8_t> &object) override;
	std::optional<compat::precompiled_ebpf_function> compile() override;
	void set_lddw_helpers(uint64_t (*map_by_fd)(uint32_t),
			      uint64_t (*map_by_idx)(uint32_t),
			      uint64_t (*map_val)(uint64_t),
			      uint64_t (*var_addr)(uint32_t),
			      uint64_t (*code_addr)(uint32_t)) override;

	class ::llvm_bpf_jit_context *get_jit_context()
	{
		return jit_ctx.get();
	}

    private:
	uint64_t (*map_by_fd)(uint32_t) = nullptr;
	uint64_t (*map_by_idx)(uint32_t) = nullptr;
	uint64_t (*map_val)(uint64_t) = nullptr;
	uint64_t (*var_addr)(uint32_t) = nullptr;
	uint64_t (*code_addr)(uint32_t) = nullptr;
	std::vector<ebpf_inst> instructions;
	std::vector<std::optional<external_function> > ext_funcs;

	std::unique_ptr<llvm_bpf_jit_context> jit_ctx;
	friend class ::llvm_bpf_jit_context;

	std::string error_msg;
	std::optional<compat::precompiled_ebpf_function> jitted_function;
};
} // namespace bpftime::vm::llvm

#endif
