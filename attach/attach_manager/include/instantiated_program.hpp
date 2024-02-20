#ifndef _BPFTIME_INSTANTIATED_PROGRAM_HPP
#define _BPFTIME_INSTANTIATED_PROGRAM_HPP
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
struct ebpf_vm;
namespace bpftime
{
namespace attach
{
class bpftime_instantiated_program {
    public:
	bpftime_instantiated_program(const void *insns, size_t insn_cnt,
				     const std::string_view name);
	virtual ~bpftime_instantiated_program();

    private:
	// The virtual machine instance
	struct ebpf_vm *vm;
	// The name of the instantiated bpftime program
	std::string prog_name;
	// The original ebpf instructions
	std::vector<uint64_t> insns;
	// Context of ufunc (bpf helpers)
	struct bpftime_ufunc_ctx *ufunc_ctx;
};
} // namespace attach
} // namespace bpftime
#endif
