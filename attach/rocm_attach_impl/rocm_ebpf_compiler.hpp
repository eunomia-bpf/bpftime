// SPDX-License-Identifier: MIT
#pragma once
#include "rocm_probe.hpp"
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <map>
namespace bpftime::attach
{
llvm::Expected<std::unique_ptr<llvm::Module>>
create_rocm_module(const rocm_probe_snapshot &, llvm::LLVMContext &,
		   uint64_t channel = 0,
		   const std::map<int, uint64_t> &maps = {});
}
