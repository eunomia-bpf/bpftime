/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _LLVM_BPF_JIT_HELPER
#define _LLVM_BPF_JIT_HELPER

#include "llvm_jit_context.hpp"
#include "ebpf_inst.h"
#include <functional>
#include <llvm/IR/Constants.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/AtomicOrdering.h>
#include <llvm/ADT/APInt.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/IRBuilder.h>
#include <map>
#include <tuple>
#include <utility>
#include <spdlog/spdlog.h>
namespace bpftime
{

static inline bool is_jmp(const ebpf_inst &insn)
{
	return (insn.code & 0x07) == EBPF_CLS_JMP ||
	       (insn.code & 0x07) == EBPF_CLS_JMP32;
}
static inline std::string ext_func_sym(uint32_t idx)
{
	char buf[32];
	sprintf(buf, "_bpf_helper_ext_%04" PRIu32, idx);
	return buf;
}

static inline bool is_alu64(const ebpf_inst &insn)
{
	return (insn.code & 0x07) == EBPF_CLS_ALU64;
}

/// Get the source representation of certain ALU operands
llvm::Value *emitLoadALUSource(const ebpf_inst &inst, llvm::Value **regs,
			       llvm::IRBuilder<> &builder);
llvm::Value *emitLoadALUDest(const ebpf_inst &inst, llvm::Value **regs,
			     llvm::IRBuilder<> &builder, bool dstAlways64);
void emitStoreALUResult(const ebpf_inst &inst, llvm::Value **regs,
			llvm::IRBuilder<> &builder, llvm::Value *result);
llvm::Expected<llvm::Value *>
emitALUEndianConversion(const ebpf_inst &inst, llvm::IRBuilder<> &builder,
			llvm::Value *dst_val);

void emitALUWithDstAndSrc(
	const ebpf_inst &inst, llvm::IRBuilder<> &builder, llvm::Value **regs,
	std::function<llvm::Value *(llvm::Value *, llvm::Value *)> func);

llvm::Value *emitStoreLoadingSrc(const ebpf_inst &inst,
				 llvm::IRBuilder<> &builder,
				 llvm::Value **regs);
void emitStoreWritingResult(const ebpf_inst &inst, llvm::IRBuilder<> &builder,
			    llvm::Value **regs, llvm::Value *result);

void emitStore(const ebpf_inst &inst, llvm::IRBuilder<> &builder,
	       llvm::Value **regs, llvm::IntegerType *destTy);

std::tuple<llvm::Value *, llvm::Value *, llvm::Value *>
emitJmpLoadSrcAndDstAndZero(const ebpf_inst &inst, llvm::Value **regs,
			    llvm::IRBuilder<> &builder);

llvm::Expected<llvm::BasicBlock *>
loadJmpDstBlock(uint16_t pc, const ebpf_inst &inst,
		const std::map<uint16_t, llvm::BasicBlock *> &instBlocks);
llvm::Expected<llvm::BasicBlock *>
loadCallDstBlock(uint16_t pc, const ebpf_inst &inst,
		 const std::map<uint16_t, llvm::BasicBlock *> &instBlocks);
llvm::Expected<llvm::BasicBlock *>
loadJmpNextBlock(uint16_t pc, const ebpf_inst &inst,
		 const std::map<uint16_t, llvm::BasicBlock *> &instBlocks);
llvm::Expected<std::pair<llvm::BasicBlock *, llvm::BasicBlock *> >
localJmpDstAndNextBlk(uint16_t pc, const ebpf_inst &inst,
		      const std::map<uint16_t, llvm::BasicBlock *> &instBlocks);
llvm::Value *emitLDXLoadingAddr(llvm::IRBuilder<> &builder, llvm::Value **regs,
				const ebpf_inst &inst);
void emitLDXStoringResult(llvm::IRBuilder<> &builder, llvm::Value **regs,
			  const ebpf_inst &inst, llvm::Value *result);
void emitLoadX(llvm::IRBuilder<> &builder, llvm::Value **regs,
	       const ebpf_inst &inst, llvm::IntegerType *srcTy);

llvm::Expected<int> emitCondJmpWithDstAndSrc(
	llvm::IRBuilder<> &builder, uint16_t pc, const ebpf_inst &inst,
	const std::map<uint16_t, llvm::BasicBlock *> &instBlocks,
	llvm::Value **regs,
	std::function<llvm::Value *(llvm::Value *, llvm::Value *)> func);

llvm::Expected<int>
emitExtFuncCall(llvm::IRBuilder<> &builder, const ebpf_inst &inst,
		const std::map<std::string, llvm::Function *> &extFunc,
		llvm::Value **regs, llvm::FunctionType *helperFuncTy,
		uint16_t pc, llvm::BasicBlock *exitBlk);
void emitAtomicBinOp(llvm::IRBuilder<> &builder, llvm::Value **regs,
		     llvm::AtomicRMWInst::BinOp op, const ebpf_inst &inst,
		     bool is64, bool is_fetch);
} // namespace bpftime
#endif
