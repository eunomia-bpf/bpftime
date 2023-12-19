#ifndef _LLVM_BPF_JIT_CONTEXT_H
#define _LLVM_BPF_JIT_CONTEXT_H

#define DEBUG_TYPE "debug"

#include "llvm_bpf_jit.h"
#include <llvm/Support/TargetSelect.h>
#include <memory>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/IR/Module.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <optional>
#include <string>
#include <pthread.h>
typedef uint8_t u8;
typedef int8_t s8;
typedef uint16_t u16;
typedef int16_t s16;
typedef uint32_t u32;
typedef int32_t s32;
typedef uint64_t u64;
typedef int64_t s64;

const static char *LDDW_HELPER_MAP_BY_FD = "__lddw_helper_map_by_fd";
const static char *LDDW_HELPER_MAP_BY_IDX = "__lddw_helper_map_by_idx";
const static char *LDDW_HELPER_MAP_VAL = "__lddw_helper_map_val";
const static char *LDDW_HELPER_VAR_ADDR = "__lddw_helper_var_addr";
const static char *LDDW_HELPER_CODE_ADDR = "__lddw_helper_code_addr";

#define IS_ALIGNED(x, a) (((uintptr_t)(x) & ((a)-1)) == 0)

struct llvm_bpf_jit_context {
	const ebpf_vm *vm;
	std::optional<std::unique_ptr<llvm::orc::LLJIT> > jit;
	std::unique_ptr<pthread_spinlock_t> compiling;
	llvm::Expected<llvm::orc::ThreadSafeModule>
	generateModule(const std::vector<std::string> &extFuncNames,
		       const std::vector<std::string> &lddwHelpers);
	void do_jit_compile();
	std::vector<uint8_t>
	do_aot_compile(const std::vector<std::string> &extFuncNames,
		       const std::vector<std::string> &lddwHelpers);

    public:
	llvm_bpf_jit_context(const ebpf_vm *m_vm);
	virtual ~llvm_bpf_jit_context();
	ebpf_jit_fn compile();
	std::vector<uint8_t> do_aot_compile();
};

#endif
