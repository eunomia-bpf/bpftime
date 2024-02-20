#include "spdlog/spdlog.h"
#include <cstring>
#include <instantiated_program.hpp>
#include <ebpf-vm.h>
#include <stdexcept>
using namespace bpftime::attach;

bpftime_instantiated_program::bpftime_instantiated_program(
	const void *insns, size_t insn_cnt, const std::string_view name)
	: prog_name(name)
{
	this->insns.resize(insn_cnt * 8);
	memcpy(this->insns.data(), insns, insn_cnt * 8);
	vm = ebpf_create();
	if (!vm) {
		SPDLOG_ERROR("Unable to create ebpf virtual machine");
		throw std::runtime_error(
			"Unable to create ebpf virtual machine");
	}
    ebpf_toggle_bounds_check(vm, false);
}
bpftime_instantiated_program::~bpftime_instantiated_program()
{
	SPDLOG_DEBUG("Destroy instantiated program: {}", name);
	ebpf_unload_code(vm);
	ebpf_destroy(vm);
}
