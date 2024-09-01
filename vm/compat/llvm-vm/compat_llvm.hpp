#ifndef _BPFTIME_VM_COMPAT_LLVM_HPP
#define _BPFTIME_VM_COMPAT_LLVM_HPP

#include <bpftime_vm_compat.hpp>
#include <memory>
#include <optional>
#include <vector>
#include <ebpf_inst.h>
#include <llvmbpf.hpp>

namespace bpftime::vm::llvm
{

class bpftime_llvm_vm : public bpftime::llvmbpf_vm, public bpftime::vm::compat::bpftime_vm_impl {
public:
    bpftime_llvm_vm() : bpftime::llvmbpf_vm() {}
    virtual ~bpftime_llvm_vm() = default;

    std::string get_error_message() override {
        return bpftime::llvmbpf_vm::get_error_message();
    }
    int load_code(const void *code, size_t code_len) override {
        return bpftime::llvmbpf_vm::load_code(code, code_len);
    }
    int register_external_function(size_t index, const std::string &name,
                                   void *fn) override;
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
};

} // namespace bpftime::vm::llvm

#endif // _BPFTIME_VM_COMPAT_LLVM_HPP
