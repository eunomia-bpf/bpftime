#include "bpftime_vm_compat.hpp"
#include <memory>

// Declare factory functions implemented in compat_*.cpp
namespace bpftime::vm::compat {
std::unique_ptr<bpftime_vm_impl> create_ubpf_vm_instance();
std::unique_ptr<bpftime_vm_impl> create_llvm_vm_instance();
}

namespace bpftime {
void bpftime_init_vm_registry()
{
    using namespace bpftime::vm::compat;
    static bool initialized = false;
    if (initialized)
        return;
    initialized = true;

#if BPFTIME_LLVM_JIT
    register_vm_factory("llvm", &create_llvm_vm_instance);
#endif
#if BPFTIME_UBPF_JIT
    register_vm_factory("ubpf", &create_ubpf_vm_instance);
#endif
}
} // namespace bpftime


