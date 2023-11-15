#ifndef _BPFTIME_PROG_HPP
#define _BPFTIME_PROG_HPP

#include <ebpf-vm.h>
#include <cinttypes>
#include <vector>
#include <string>
namespace bpftime
{

// executable program for bpf function
class bpftime_prog {
    public:
	enum class bpftime_vm_type {
		VM,
		JIT,
		KERNEL,
	};

	const char *prog_name() const
	{
		return name.c_str();
	}
	bpftime_prog(const ebpf_inst *insn, size_t insn_cnt, const char *name);
	~bpftime_prog();

	// load the programs to userspace vm or compile the jit program
	// if program_name is NULL, will load the first program in the object
	int bpftime_prog_load(bool jit_enabled);
	int bpftime_prog_load(bpftime_vm_type type);
	int bpftime_prog_unload();

	// exec in user space or kernel space
	int bpftime_prog_exec(void *memory, size_t memory_size,
			      uint64_t *return_val) const;
	int bpftime_prog_register_raw_helper(struct bpftime_helper_info info);
	const std::vector<ebpf_inst> &get_insns() const
	{
		return insns;
	}

    private:
	bpftime_vm_type vm_type;
	int bpftime_prog_set_insn(struct ebpf_inst *insn, size_t insn_cnt);
	std::string name;
	// vm at the first element
	struct ebpf_vm *vm;

	// used in jit
	ebpf_jit_fn fn;
	std::vector<struct ebpf_inst> insns;

	char *errmsg;

	// ffi ctx
	struct bpftime_ffi_ctx *ffi_ctx;

	// kernel runtime
	int prog_fd;
};

} // namespace bpftime
#endif
