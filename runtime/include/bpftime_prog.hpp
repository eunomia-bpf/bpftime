/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_PROG_HPP
#define _BPFTIME_PROG_HPP

#include <ebpf-vm.h>
#include <optional>
#include <vector>
#include <string>
namespace bpftime
{

extern thread_local std::optional<uint64_t> current_thread_bpf_cookie;

// executable program for bpf function
class bpftime_prog {
    public:
	const char *prog_name() const
	{
		return name.c_str();
	}
	bpftime_prog(const ebpf_inst *insn, size_t insn_cnt, const char *name);
	~bpftime_prog();

	// load the programs to userspace vm or compile the jit program
	// if program_name is NULL, will load the first program in the object
	int bpftime_prog_load(bool jit);
	int bpftime_prog_unload();

	// exec in user space
	int bpftime_prog_exec(void *memory, size_t memory_size,
			      uint64_t *return_val) const;

	int bpftime_prog_register_raw_helper(struct bpftime_helper_info info);
	const std::vector<ebpf_inst> &get_insns() const
	{
		return insns;
	}
	const struct ebpf_vm *get_vm() const { return vm; }
	int load_aot_object(const std::vector<uint8_t> &buf);
    private:
	int bpftime_prog_set_insn(struct ebpf_inst *insn, size_t insn_cnt);
	std::string name;
	// vm at the first element
	struct ebpf_vm *vm;

	bool jitted;

	// used in jit
	ebpf_jit_fn fn;
	std::vector<struct ebpf_inst> insns;

	char *errmsg;

	// ufunc ctx
	struct bpftime_ufunc_ctx *ufunc_ctx;

	// kernel runtime
};

} // namespace bpftime
#endif
