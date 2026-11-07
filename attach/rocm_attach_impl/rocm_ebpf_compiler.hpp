// SPDX-License-Identifier: MIT
#pragma once
#include "rocm_probe.hpp"
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
namespace bpftime::attach
{
llvm::Expected<std::unique_ptr<llvm::Module>>
create_rocm_module(const rocm_probe_snapshot &, llvm::LLVMContext &);
}
