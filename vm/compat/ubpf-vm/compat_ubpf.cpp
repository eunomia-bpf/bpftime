#include "ebpf_inst.h"
#include "spdlog/spdlog.h"
#include <cerrno>
#include <memory>
#include <stdexcept>
#include <string>
#include <ubpf.h>
#include <bpftime_vm_compat.hpp>
#include <compat_ubpf.hpp>
#include <vector>

namespace bpftime::vm::compat
{

std::unique_ptr<bpftime_vm_impl> create_vm_instance()
{
	return std::make_unique<ubpf::bpftime_ubpf_vm>();
}

} // namespace bpftime::vm::compat

using namespace bpftime::vm::ubpf;

bpftime_ubpf_vm::bpftime_ubpf_vm()
{
	ubpf_vm = ubpf_create();
	if (!ubpf_vm) {
		throw std::runtime_error(
			"Unable to create ubpf virtual machine");
	}
}

std::string bpftime_ubpf_vm::get_error_message()
{
	return error_string;
}

bool bpftime_ubpf_vm::toggle_bounds_check(bool enable)
{
	return ubpf_toggle_bounds_check(ubpf_vm, enable);
}

void bpftime_ubpf_vm::register_error_print_callback(int (*fn)(FILE *,
							      const char *,
							      ...))
{
	ubpf_set_error_print(ubpf_vm, fn);
}

int bpftime_ubpf_vm::register_external_function(size_t index,
						const std::string &name,
						void *fn)
{
	// Allocate one more id
	size_t next_id = next_helper_id++;
	helper_id_map[index] = next_id;
	return ubpf_register(ubpf_vm, next_id, name.c_str(),
			     (external_function_t)fn);
}

int bpftime_ubpf_vm::load_code(const void *code, size_t code_len)
{
	SPDLOG_DEBUG("Loading instructions of ubpf vm with length {}",
		     code_len);
	if (code_len % 8 != 0) {
		error_string = "Length of code must be a multiple of 8";
		return -1;
	}
	auto insn_count = code_len / 8;
	std::vector<ebpf_inst> insns((ebpf_inst *)code,
				     (ebpf_inst *)code + insn_count);
	for (size_t i = 0; i < insn_count; i++) {
		auto &curr_insn = insns[i];
		// Patch helper call
		if (curr_insn.code == EBPF_OP_CALL) {
			if (auto itr = helper_id_map.find(curr_insn.imm);
			    itr != helper_id_map.end()) {
				curr_insn.imm = itr->second;
				SPDLOG_DEBUG(
					"Patched call at pc {}, helper {} to {}",
					i, curr_insn.imm, itr->second);
			} else {
				if (curr_insn.imm >= 64) {
					error_string =
						"invalid call immediate at PC " +
						std::to_string(i);
					return -EINVAL;
				} else {
					error_string =
						"call to nonexistent function " +
						std::to_string(curr_insn.imm) +
						" at PC " + std::to_string(i);
					return -EINVAL;
				}
			}
		} else
			// Patch LDDW
			if (curr_insn.code == EBPF_OP_LDDW) {
				if (i + 1 == insn_count) {
					error_string =
						"Unable to patch lddw instructions at " +
						std::to_string(i) +
						", it's the last instruction";
					return -EINVAL;
				}
				SPDLOG_DEBUG("Checking lddw at pc {}", i);
				auto &next_insn = insns[i + 1];
				uint64_t imm;
				// Patch lddw instructions..
				if (curr_insn.src_reg == 0) {
					imm = ((uint64_t)(uint32_t)
						       curr_insn.imm) |
					      ((uint64_t)(uint32_t)next_insn.imm
					       << 32);
				} else if (curr_insn.src_reg == 1) {
					if (!map_by_fd) {
						error_string =
							"Unable to patch lddw instruction at " +
							std::to_string(i) +
							", map_by_fd not defined";
						return -EINVAL;
					}
					imm = map_by_fd(curr_insn.imm);
				} else if (curr_insn.src_reg == 2) {
					if (!map_by_fd || !map_val) {
						error_string =
							"Unable to patch lddw instruction at " +
							std::to_string(i) +
							", map_by_fd or map_val not defined";
						return -EINVAL;
					}
					imm = map_val(map_by_fd(
						      curr_insn.imm)) +
					      next_insn.imm;
				} else if (curr_insn.src_reg == 3) {
					if (!var_addr) {
						error_string =
							"Unable to patch lddw instruction at " +
							std::to_string(i) +
							", var_addr not defined";
						return -EINVAL;
					}
					imm = var_addr(curr_insn.imm);
				} else if (curr_insn.src_reg == 4) {
					if (!code_addr) {
						error_string =
							"Unable to patch lddw instruction at " +
							std::to_string(i) +
							", code_addr not defined";
						return -EINVAL;
					}
					imm = code_addr(curr_insn.imm);
				} else if (curr_insn.src_reg == 5) {
					if (!map_by_idx) {
						error_string =
							"Unable to patch lddw instruction at " +
							std::to_string(i) +
							", map_by_idx not defined";
						return -EINVAL;
					}
					imm = map_by_idx(curr_insn.imm);
				} else if (curr_insn.src_reg == 6) {
					if (!map_by_idx || !map_val) {
						error_string =
							"Unable to patch lddw instruction at " +
							std::to_string(i) +
							", map_by_idx or map_val not defined";
						return -EINVAL;
					}
					imm = map_val(map_by_idx(
						      curr_insn.imm)) +
					      next_insn.imm;
				} else {
					error_string =
						"Unable to patch lddw instruction at " +
						std::to_string(i) +
						", unsupported src_reg " +
						std::to_string(
							curr_insn.src_reg);

					return -EINVAL;
				}
				SPDLOG_DEBUG(
					"Patched lddw instruction at pc {}, imm={:x}",
					i, imm);
				curr_insn.imm = imm & 0xffffffff;
				next_insn.imm = imm >> 32;
				curr_insn.src_reg = 0;
				i++;
			}
	}
	char *errmsg = nullptr;
	int err;
	if (err = ubpf_load(ubpf_vm, insns.data(), insns.size() * 8, &errmsg);
	    err < 0) {
		error_string = errmsg;
		free(errmsg);
	}
	return err;
}

void bpftime_ubpf_vm::unload_code()
{
	ubpf_unload_code(ubpf_vm);
}

int bpftime_ubpf_vm::exec(void *mem, size_t mem_len, uint64_t &bpf_return_value)
{
	return ubpf_exec(ubpf_vm, mem, mem_len, &bpf_return_value);
}

std::optional<bpftime::vm::compat::precompiled_ebpf_function>
bpftime_ubpf_vm::compile()
{
	char *errmsg = nullptr;
	auto result = ubpf_compile(ubpf_vm, &errmsg);
	if (!result) {
		error_string = errmsg;
		free(errmsg);
		SPDLOG_ERROR("Failed to compile using ubpf: {}", error_string);
		return {};
	}
	return result;
}

void bpftime_ubpf_vm::set_lddw_helpers(uint64_t (*map_by_fd)(uint32_t),
				       uint64_t (*map_by_idx)(uint32_t),
				       uint64_t (*map_val)(uint64_t),
				       uint64_t (*var_addr)(uint32_t),
				       uint64_t (*code_addr)(uint32_t))
{
	this->map_by_fd = map_by_fd;
	this->map_by_idx = map_by_idx;
	this->map_val = map_val;
	this->var_addr = var_addr;
	this->code_addr = code_addr;
}

int bpftime_ubpf_vm::set_unwind_function_index(size_t idx)
{
	return ubpf_set_unwind_function_index(ubpf_vm, idx);
}

int bpftime_ubpf_vm::set_pointer_secret(uint64_t secret)
{
	return ubpf_set_pointer_secret(ubpf_vm, secret);
}
