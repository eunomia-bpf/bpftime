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

class bpftime_llvm_vm : public bpftime::vm::compat::bpftime_vm_impl,
			    public bpftime::llvmbpf_vm {
	bpftime_llvm_vm() : bpftime::llvmbpf_vm()
	{
	}
};
} // namespace bpftime::vm::llvm

#endif
