/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <handlers/prog_handler.hpp>
namespace bpftime
{
namespace shm_common
{
bpf_prog_handler::bpf_prog_handler(managed_shared_memory &mem, const void *insn,
				   size_t insn_cnt, const char *prog_name,
				   int prog_type)
	: insns(shm_ebpf_inst_vector_allocator(mem.get_segment_manager())),
	  attach_fds(shm_pair_vector_allocator(mem.get_segment_manager())),
	  name(char_allocator(mem.get_segment_manager()))
{
	insns.assign((uint64_t *)insn, (uint64_t *)insn + insn_cnt);
	this->name = prog_name;
}
} // namespace shm_common
} // namespace bpftime
