/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime.hpp"
#include "base_attach_impl.hpp"
#include "bpftime_config.hpp"
#include "bpftime_helper_group.hpp"
#include "bpftime_internal.h"
#include "bpftime_shm_internal.hpp"
#include "bpftime_vm_compat.hpp"
#include "platform_utils.hpp"
#include "ebpf-vm.h"
#ifdef BPFTIME_ENABLE_CUDA_ATTACH
#include "nvPTXCompiler.h"
#endif
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>

#if defined(__x86_64__)
// The pinned uBPF x86_64 JIT emulates fetch-ALU atomics (add/or/and/xor with
// BPF_FETCH) with a cmpxchg retry loop. When the destination register is r0
// (mapped to native RAX), the initial load overwrites RAX with the old memory
// value and the retried cmpxchg dereferences that value as the address,
// crashing the host process. The interpreter and the aarch64 JIT are correct.
// Until the defect is fixed upstream, keep the historical fail-closed
// contract for this single encoding by steering it to the interpreter.
static bool
ubpf_x86_jit_unsafe_fetch_into_r0(const struct ebpf_inst *insn, size_t cnt)
{
	for (size_t i = 0; i < cnt; i++) {
		const struct ebpf_inst &in = insn[i];
		if (in.code != EBPF_ATOMIC_OPCODE_32 &&
		    in.code != EBPF_ATOMIC_OPCODE_64)
			continue;
		if (in.dst_reg != 0)
			continue;
		// Only fetch-ALU is affected; non-fetch, XCHG and CMPXCHG use
		// safe single-instruction sequences.
		if (!(in.imm & EBPF_FETCH))
			continue;
		if (in.imm == EBPF_XCHG || in.imm == EBPF_CMPXCHG)
			continue;
		return true;
	}
	return false;
}
#endif

#include <atomic>
#include <chrono>
#include <functional>
#include <spdlog/spdlog.h>
#include <stdexcept>
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
namespace
{
class bpf_cpu_execution_guard {
    public:
	bpf_cpu_execution_guard()
	{
		if (bpftime_get_current_bpf_cpu() >= 0)
			return;
		// Map-free programs may use only process-local runtime config.
		if (!shm_holder.global_shared_memory.get_manager())
			return;
		cpu = bpftime_get_native_cpu();
		if (cpu < 0) {
			warn_throttled([] { SPDLOG_WARN("sched_getcpu failed; BPF execution skipped"); });
			return;
		}
		if (cpu >= BPFTIME_MAX_CPU_COUNT) {
			// `cpu` is a member, so it cannot be captured by a lambda;
			// copy it into a local first.
			const int cpu_for_log = cpu;
			warn_throttled([cpu_for_log] {
				SPDLOG_WARN(
					"logical cpu {} exceeds BPFTIME_MAX_CPU_COUNT={}; BPF execution skipped",
					cpu_for_log, BPFTIME_MAX_CPU_COUNT);
			});
			return;
		}
		if (!shm_holder.global_shared_memory.acquire_bpf_cpu(cpu)) {
			const int cpu_for_log = cpu;
			warn_throttled([cpu_for_log] {
				SPDLOG_WARN(
					"per-CPU lock for cpu {} unavailable; BPF execution skipped",
					cpu_for_log);
			});
			return;
		}
		bpftime_set_current_bpf_cpu(cpu);
		owns_lock = true;
	}

	~bpf_cpu_execution_guard() { release(); }

	bool entered() const
	{
		return owns_lock || bpftime_get_current_bpf_cpu() >= 0 ||
		       !shm_holder.global_shared_memory.get_manager();
	}

	void release()
	{
		if (!owns_lock)
			return;
		bpftime_set_current_bpf_cpu(-1);
		shm_holder.global_shared_memory.unlock_bpf_cpu(cpu);
		owns_lock = false;
	}

    private:
	// Rate-limit degradation warnings: a hot BPF path must not spam the log on
	// every event when the environment is outside the supported CPU range.
	// Shared 5s rate limiter; the emit callable runs SPDLOG_WARN only when admitted.
	template <class F> static void warn_throttled(F &&emit)
	{
		static std::atomic<std::int64_t> last_log_ns{0};
		constexpr std::int64_t MIN_INTERVAL_NS = 5'000'000'000LL;
		std::int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
				   std::chrono::steady_clock::now().time_since_epoch())
				   .count();
		std::int64_t prev = last_log_ns.load(std::memory_order_relaxed);
		if (now - prev < MIN_INTERVAL_NS)
			return;
		if (!last_log_ns.compare_exchange_strong(prev, now,
							      std::memory_order_relaxed))
			return;
		emit();
	}

	int cpu = -1;
	bool owns_lock = false;
};

} // namespace

#ifdef BPFTIME_ENABLE_CUDA_ATTACH
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
#endif
thread_local std::optional<uint64_t> current_thread_bpf_cookie;
thread_local std::optional<uint64_t> current_thread_tail_call_ret;
thread_local std::optional<exec_context_info> current_thread_exec_ctx;

bpftime_prog::bpftime_prog(const struct ebpf_inst *insn, size_t insn_cnt,
			   const char *name)
	: name(name)
{
	SPDLOG_DEBUG("Creating bpftime_prog with name {}", name);
	insns.assign(insn, insn + insn_cnt);
	const char *vm_name = bpftime::bpftime_get_runtime_config().get_vm_name();
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
			   const char *name, runtime_config config)
	: name(name)
{
	// BPFtime_prog relies on the global shared memory being properly
	// initialized to function.
	SPDLOG_DEBUG("Creating bpftime_prog with name {}", name);
	insns.assign(insn, insn + insn_cnt);
	bpftime::bpftime_set_runtime_config(std::move(config));
	const char *vm_name = bpftime::bpftime_get_runtime_config().get_vm_name();
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
	signal_ready = false;
	loaded_from_aot = false;
	program_loaded = false;
	int res = -1;

	SPDLOG_DEBUG("Load insn cnt {}", insns.size());
	res = ebpf_load(vm, insns.data(),
			insns.size() * sizeof(struct ebpf_inst), &errmsg);
	if (res < 0) {
		SPDLOG_ERROR("Failed to load insn: {}", errmsg);
		return res;
	}

	// Reject programs that call helpers which were never registered into
	// this VM. The LLVM JIT used to fail late and ambiguously: compilation
	// returned "Ext func not found: _bpf_helper_ext_XXXX", the failure was
	// demoted to a warning, jitted was set to false and the load still
	// succeeded, so every invocation died at runtime with "Unable to run
	// ebpf callback" instead of a load-time error (see #704). Local calls
	// (call) are handled by the VM itself and are not audited here,
	// mirroring prepare_for_signal_execution().
	//
	// CUDA programs (cuda__*) are exempt: their helpers (e.g. 503 =
	// bpf_get_block_idx) are resolved on the device side. They are never
	// registered into this host VM; ptxpass generates device PTX from an
	// isolated VM with its own helper table, and the runtime trampoline
	// provides the actual implementations (e.g. _bpf_helper_ext_0503).
	// This VM is only used here to produce that device code, so the
	// host-side audit does not apply.
	if (!is_cuda()) {
		for (size_t i = 0; i < insns.size(); ++i) {
			const struct ebpf_inst &inst = insns[i];
			if (inst.code == 0x18) {
				// Second instruction of a LDDW is data, not code.
				++i;
				continue;
			}
			if (inst.code == 0x8d) // Indirect CALLX cannot be audited here.
				break;
			if (inst.code != 0x85 || inst.src_reg != 0)
				continue;
			if (registered_helper_ids.count(
				    static_cast<unsigned>(inst.imm)) == 0) {
				SPDLOG_ERROR(
					"Program {} calls helper {} which is not registered in the VM. Register the helper (e.g. bpftime_helper_group::add_helper_group_to_prog) or check the bytecode before loading.",
					name, inst.imm);
				return -EINVAL;
			}
		}
	}
	if (jit) {
#if defined(__x86_64__)
		// The defect is in the uBPF x86_64 JIT only; the LLVM backend
		// handles r0 atomics correctly, so do not change its path.
		const char *cur_vm_name =
			bpftime::bpftime_get_runtime_config()
				.get_vm_name();
		if (std::string(cur_vm_name) == "ubpf" &&
		    ubpf_x86_jit_unsafe_fetch_into_r0(insns.data(),
							      insns.size())) {
			SPDLOG_WARN("uBPF x86_64 JIT cannot safely compile a fetch-atomic targeting r0; using the interpreter instead");
			jitted = false;
		} else
#endif
		{
			ebpf_jit_fn jit_fn = ebpf_compile(vm, &errmsg);
			if (jit_fn == NULL) {
				SPDLOG_WARN("JIT compilation failed ({}), falling back to interpreter", errmsg);
				jitted = false;
			} else {
				jitted = true;
				fn = jit_fn;
			}
		}
	} else {
		jitted = false;
	}

	program_loaded = true;
	return 0;
}

int bpftime_prog::bpftime_prog_unload()
{
	signal_ready = false;
	program_loaded = false;
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
	if (attach::current_signal_callback) {
		if (!signal_ready || !jitted || !return_val)
			return -ENOTSUP;
		// No VM dispatcher, lazy compilation, logging, dynamic TLS or
		// shared-memory protection helpers may run from SIGTRAP.
		*return_val = fn(memory, memory_size);
		return 0;
	}
	// Publish the caller-provided context pointer and size so helpers
	// that copy the context (e.g. bpftime_tail_call) can tell the
	// host-owned entry context from a program-built replacement context.
	// Tail calls run nested exec calls, so save and restore the outer
	// value to keep the depth correct.
	struct exec_ctx_guard {
		std::optional<exec_context_info> saved_ctx;
		explicit exec_ctx_guard(const void *ptr, size_t size)
			: saved_ctx(current_thread_exec_ctx)
		{
			current_thread_exec_ctx = exec_context_info{
				ptr, size
			};
		}
		~exec_ctx_guard()
		{
			current_thread_exec_ctx = saved_ctx;
		}
	} exec_ctx_guard(memory, memory_size);
	if (is_cuda()) {
		throw std::runtime_error("Unable to execute CUDA program");
	}
	uint64_t val = 0;
	int res = 0;
	// set share memory read and write able
	bpftime_protect_disable();
	bpf_cpu_execution_guard cpu_guard;
	if (!cpu_guard.entered()) {
		bpftime_protect_enable();
		return -EAGAIN;
	}
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
	*return_val = current_thread_tail_call_ret.value_or(val);
	current_thread_tail_call_ret.reset();
	cpu_guard.release();
	// set share memory read only
	bpftime_protect_enable();
	return res;
}

int bpftime_prog::bpftime_prog_register_raw_helper(
	struct bpftime_helper_info info)
{
	int result = ebpf_register(vm, info.index, info.name.c_str(), info.fn);
	if (result == 0) {
		signal_ready = false;
		signal_safe_helpers[info.index] = helper_is_async_signal_safe(info.index, info.fn);
		registered_helper_ids.insert(info.index);
	}
	return result;
}

int bpftime_prog::prepare_for_signal_execution()
{
	// A program can be linked to several live sites. Once prepared, adding
	// another link must not mutate state read by an existing signal callback.
	if (signal_ready)
		return 0;
	if (!program_loaded)
		return -EINVAL;
#if BPFTIME_ENABLE_MPK
	// The MPK access transition has not been audited for signal callbacks.
	return -ENOTSUP;
#endif
	if (is_cuda() || loaded_from_aot)
		return -ENOTSUP;
	const char *backend = ebpf_get_vm_name(vm);
	if (strcmp(backend, "llvm") != 0 && strcmp(backend, "ubpf") != 0)
		return -ENOTSUP;
	for (size_t i = 0; i < insns.size(); ++i) {
		const auto &inst = insns[i];
		if (inst.code == 0x18) { // LDDW's second word is data.
			++i;
			continue;
		}
		if (inst.code == 0x8d) // Indirect CALLX cannot be audited here.
			return -ENOTSUP;
		if (inst.code != 0x85)
			continue;
		if (inst.src_reg == 1) {
			// Only LLVM's compatibility layer distinguishes local calls
			// from helper calls when loading bytecode.
			if (strcmp(backend, "llvm") == 0)
				continue;
			return -ENOTSUP;
		}
		auto helper = signal_safe_helpers.find((unsigned)inst.imm);
		if (inst.src_reg != 0 || helper == signal_safe_helpers.end() ||
		    !helper->second) {
			SPDLOG_ERROR("Trap attach rejected for {}: helper {} is not async-signal-safe",
				     name, inst.imm);
			return -ENOTSUP;
		}
	}
#if defined(__x86_64__)
	// The uBPF x86 JIT corrupts RAX when a fetch-ALU atomic uses r0 as
	// its memory base. The ordinary load path falls back to the interpreter,
	// but a trap callback must be JIT-compiled. Reject this attach rather
	// than compiling host-crashing code in the signal-admission path.
	if (strcmp(backend, "ubpf") == 0 &&
	    ubpf_x86_jit_unsafe_fetch_into_r0(insns.data(), insns.size())) {
		SPDLOG_WARN("Rejecting trap attach: uBPF x86 JIT cannot safely compile a fetch-atomic targeting r0");
		return -ENOTSUP;
	}
#endif
	if (!jitted) {
		auto compiled = ebpf_compile(vm, &errmsg);
		if (!compiled)
			return -ENOTSUP;
		fn = compiled;
		jitted = true;
	}
	signal_ready = true;
	return 0;
}

int bpftime_prog::load_aot_object(const std::vector<uint8_t> &buf)
{
	signal_ready = false;
	loaded_from_aot = true;
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
