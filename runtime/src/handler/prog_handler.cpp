/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <handler/prog_handler.hpp>
namespace bpftime
{
bpf_prog_handler::bpf_prog_handler(managed_shared_memory &mem,
				   const struct ebpf_inst *insn,
				   size_t insn_cnt, const char *prog_name,
				   int prog_type)
	: insns(shm_ebpf_inst_vector_allocator(mem.get_segment_manager())),
	aot_insns(shm_aot_inst_vector_allocator(mem.get_segment_manager())),
	  name(char_allocator(mem.get_segment_manager()))
{
	insns.assign(insn, insn + insn_cnt);
	this->name = prog_name;
}
} // namespace bpftime
