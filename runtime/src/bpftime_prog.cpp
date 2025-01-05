/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime.hpp"
#include "bpftime_config.hpp"
#include "bpftime_helper_group.hpp"
#include "bpftime_internal.h"
#include "bpftime_vm_compat.hpp"
#include "ebpf-vm.h"
#include "llvm_jit_context.hpp"
#include "nvPTXCompiler.h"
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include "bpftime_vm_compat.hpp"
#include <cstdio>

#define NVPTXCOMPILER_SAFE_CALL(x)                                             \
	do {                                                                   \
		nvPTXCompileResult result = x;                                 \
		if (result != NVPTXCOMPILE_SUCCESS) {                          \
			SPDLOG_ERROR("error: {} failed with error code {}",    \
				     #x, (int)result);                         \
			return {};                                             \
		}                                                              \
	} while (0)

using namespace std;
namespace bpftime
{
std::optional<std::vector<char>> compile_ptx_to_elf(const std::string &ptx_code,
						    const char *cpu_target)
{
	unsigned int minor_version, major_version;
	NVPTXCOMPILER_SAFE_CALL(
		nvPTXCompilerGetVersion(&major_version, &minor_version));
	SPDLOG_INFO("ptx compiler version: {}.{}", major_version,
		    minor_version);
	SPDLOG_DEBUG("Starting compiling with ptx code: \n{}", ptx_code);
	nvPTXCompilerHandle handle = nullptr;
	NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerCreate(
		&handle, (size_t)ptx_code.size(), ptx_code.c_str()));
	const auto deleter = [](struct nvPTXCompiler *ptr) {
		if (auto err = nvPTXCompilerDestroy(&ptr);
		    err != NVPTXCOMPILE_SUCCESS) {
			SPDLOG_ERROR("Unable to destroy compiler: {}",
				     (int)err);
		}
	};
	std::unique_ptr<struct nvPTXCompiler, decltype(deleter)> compiler(
		handle, deleter);
	std::string opt1 = "--gpu-name=";
	opt1 += cpu_target;
	const char *compile_options[] = { opt1.c_str(), "--verbose" };

	if (auto result = nvPTXCompilerCompile(compiler.get(),
					       std::size(compile_options),
					       compile_options);
	    result != NVPTXCOMPILE_SUCCESS) {
		size_t sz;
		NVPTXCOMPILER_SAFE_CALL(
			nvPTXCompilerGetErrorLogSize(compiler.get(), &sz));
		std::string error(sz, 0);
		if (sz != 0) {
			NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLog(
				compiler.get(), (char *)error.c_str()));
			SPDLOG_ERROR("Unable to compile ptx: {}", error);
			return {};
		}
	}
	size_t elf_size;
	NVPTXCOMPILER_SAFE_CALL(
		nvPTXCompilerGetCompiledProgramSize(compiler.get(), &elf_size));
	std::vector<char> elf_binary(elf_size, 0);
	NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgram(
		compiler.get(), (void *)elf_binary.data()));

	{
		size_t info_size;

		NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLogSize(
			compiler.get(), &info_size));
		std::string info(info_size, 0);
		if (info_size != 0) {
			NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLog(
				compiler.get(), (char *)info.c_str()));
			SPDLOG_INFO("Compiler log: {}", info);
		}
	}

	return elf_binary;
}

thread_local std::optional<uint64_t> current_thread_bpf_cookie;

bpftime_prog::bpftime_prog(const struct ebpf_inst *insn, size_t insn_cnt,
			   const char *name)
	: name(name)
{
	SPDLOG_DEBUG("Creating bpftime_prog with name {}", name);
	insns.assign(insn, insn + insn_cnt);
	const char *vm_name = bpftime::bpftime_get_agent_config().get_vm_name();
	std::string vm_name_str = (std::string)vm_name;

	if (vm_name_str == "llvm") {
		SPDLOG_DEBUG("Creating vm with name {}", vm_name_str);
	} else if (vm_name_str == "ubpf") {
		SPDLOG_DEBUG("Creating vm with name {}", vm_name_str);
	} else {
		SPDLOG_DEBUG("Trying enabling non-builtin vm {}", vm_name_str);
	}

	vm = ebpf_create(vm_name);
	// Disable bounds check because we have no implementation yet
	// ebpf_toggle_bounds_check(vm, false);
	ebpf_set_lddw_helpers(vm, map_ptr_by_fd, nullptr, map_val, nullptr,
			      nullptr);
}

bpftime_prog::bpftime_prog(const struct ebpf_inst *insn, size_t insn_cnt,
			   const char *name, agent_config config)
	: name(name)
{
	// BPFtime_prog relies on the global shared memory being properly
	// initialized to function.
	SPDLOG_DEBUG("Creating bpftime_prog with name {}", name);
	insns.assign(insn, insn + insn_cnt);
	bpftime::bpftime_set_agent_config(std::move(config));
	const char *vm_name = bpftime::bpftime_get_agent_config().get_vm_name();
	std::string vm_name_str = (std::string)vm_name;

	if (vm_name_str == "llvm") {
		SPDLOG_DEBUG("Creating vm with name {}", vm_name_str);
	} else if (vm_name_str == "ubpf") {
		SPDLOG_DEBUG("Creating vm with name {}", vm_name_str);
	} else {
		SPDLOG_DEBUG("Trying enabling non-builtin vm {}", vm_name_str);
	}

	vm = ebpf_create(vm_name);
	// Disable bounds check because we have no implementation yet
	// ebpf_toggle_bounds_check(vm, false);
	if (!is_cuda()) {
		ebpf_set_lddw_helpers(vm, map_ptr_by_fd, nullptr, map_val,
				      nullptr, nullptr);
	} else {
		SPDLOG_INFO("Do not set lddw helpers due to cuda program");
		ebpf_set_lddw_helpers(vm, map_ptr_by_fd, nullptr, map_val,
				      nullptr, nullptr);
	}
}

bpftime_prog::~bpftime_prog()
{
	ebpf_unload_code(vm);
	ebpf_destroy(vm);
}

int bpftime_prog::bpftime_prog_load(bool jit)
{
	int res = -1;

	SPDLOG_DEBUG("Load insn cnt {}", insns.size());
	res = ebpf_load(vm, insns.data(),
			insns.size() * sizeof(struct ebpf_inst), &errmsg);
	if (res < 0) {
		SPDLOG_ERROR("Failed to load insn: {}", errmsg);
		return res;
	}
	if (is_cuda()) {
		// SPDLOG_INFO("Compiling CUDA program");

		// this->bpftime_prog_register_raw_helper(bpftime_helper_info{
		// 	.index = 501,
		// 	.name = "puts",
		// 	.fn = (void *)&puts,
		// });

		// ptx_code = ((struct ebpf_vm *)vm)
		// 		   ->vm_instance->generate_ptx("sm_60");

		// if (!ptx_code.has_value()) {
		// 	throw std::runtime_error("Failed to generate ptx code");
		// }
		// *ptx_code =
		// 	wrap_ptx_with_trampoline(patch_helper_names_and_header(
		// 		patch_main_from_func_to_entry(*ptx_code)));
		// cuda_elf_binary = compile_ptx_to_elf(*ptx_code, "sm_60");
		// if (!cuda_elf_binary.has_value()) {
		// 	throw std::runtime_error("unable to compile ptx code");
		// }
	} else {
		if (jit) {
			// run with jit mode
			jitted = true;
			ebpf_jit_fn jit_fn = ebpf_compile(vm, &errmsg);
			if (jit_fn == NULL) {
				SPDLOG_ERROR("Failed to compile: {}", errmsg);
				return -1;
			}
			fn = jit_fn;
		} else {
			// ignore for vm
			jitted = false;
		}
	}

	return 0;
}

int bpftime_prog::bpftime_prog_unload()
{
	if (jitted) {
		// ignore for jit
		return 0;
	}
	ebpf_unload_code(vm);
	return 0;
}

int bpftime_prog::bpftime_prog_exec(void *memory, size_t memory_size,
				    uint64_t *return_val) const
{
	if (is_cuda()) {
		throw std::runtime_error("Unable to execute CUDA program");
	}
	uint64_t val = 0;
	int res = 0;
	// set share memory read and write able
	bpftime_protect_disable();
	SPDLOG_DEBUG(
		"Calling bpftime_prog::bpftime_prog_exec, memory={:x}, memory_size={}, return_val={:x}, prog_name={}",
		(uintptr_t)memory, memory_size, (uintptr_t)return_val,
		this->name);
	if (jitted) {
		SPDLOG_DEBUG("Directly call jitted function at {:x}",
			     (uintptr_t)fn);
		val = fn(memory, memory_size);
	} else {
		SPDLOG_DEBUG("Running using ebpf_exec");
		res = ebpf_exec(vm, memory, memory_size, &val);
		if (res < 0) {
			SPDLOG_ERROR("ebpf_exec returned error: {}", res);
		}
	}
	*return_val = val;
	// set share memory read only
	bpftime_protect_enable();
	return res;
}

int bpftime_prog::bpftime_prog_register_raw_helper(
	struct bpftime_helper_info info)
{
	return ebpf_register(vm, info.index, info.name.c_str(), info.fn);
}

int bpftime_prog::load_aot_object(const std::vector<uint8_t> &buf)
{
	ebpf_jit_fn res = ebpf_load_aot_object(vm, buf.data(), buf.size());
	if (res == nullptr) {
		SPDLOG_ERROR("Failed to load aot object");
		return -1;
	}
	fn = res;
	jitted = true;
	return 0;
}

} // namespace bpftime
