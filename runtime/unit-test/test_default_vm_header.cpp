#include <bpftime_config.hpp>
#include <string_view>

#if defined(BPFTIME_LLVM_JIT)
constexpr auto expected_vm = "llvm";
#elif defined(BPFTIME_UBPF_JIT)
constexpr auto expected_vm = "ubpf";
#else
constexpr auto expected_vm = "";
#endif
static_assert(std::string_view(DEFAULT_VM_NAME) == expected_vm);
