#include "ebpf_inst.h"
#include "llvm_bpf_jit.h"
#include "llvm_jit_context.h"
#include "bpf_jit_helpers.h"
#include "spdlog/spdlog.h"
#include <iterator>

#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm-15/llvm/ExecutionEngine/JITSymbol.h>
#include <llvm-15/llvm/Support/raw_ostream.h>
#include <llvm/Support/Error.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include <utility>
#include <iostream>
#include <string>
#include <cinttypes>
#include <spdlog/spdlog.h>
using namespace llvm;
using namespace llvm::orc;

ExitOnError ExitOnErr;

static void optimizeModule(llvm::Module &M)
{
	llvm::legacy::PassManager PM;

	llvm::PassManagerBuilder PMB;
	PMB.OptLevel = 3;
	PMB.populateModulePassManager(PM);

	PM.run(M);
}

#if defined(__arm__) || defined(_M_ARM)
extern "C" void __aeabi_unwind_cpp_pr1();
#endif

ebpf_jit_fn bpf_jit_context::compile()
{
	spdlog::info("Compiling using LLJIT");
	// Create a JIT builder
	auto jit = ExitOnErr(LLJITBuilder().create());
	auto &mainDylib = jit->getMainJITDylib();
	std::vector<std::string> extFuncNames;
	// insert the helper functions
	SymbolMap extSymbols;
	for (uint32_t i = 0; i < std::size(vm->ext_funcs); i++) {
		if (vm->ext_funcs[i] != nullptr) {
			auto sym = JITEvaluatedSymbol::fromPointer(
				vm->ext_funcs[i]);
			auto symName = jit->mangleAndIntern(ext_func_sym(i));
			sym.setFlags(JITSymbolFlags::Callable |
				     JITSymbolFlags::Exported);
			extSymbols.try_emplace(symName, sym);
			extFuncNames.push_back(ext_func_sym(i));
		}
	}
#if defined(__arm__) || defined(_M_ARM)
	spdlog::info("Defining __aeabi_unwind_cpp_pr1 on arm32");
	extSymbols.try_emplace(
		jit->mangleAndIntern("__aeabi_unwind_cpp_pr1"),
		JITEvaluatedSymbol::fromPointer(__aeabi_unwind_cpp_pr1));
#endif
	ExitOnErr(mainDylib.define(absoluteSymbols(extSymbols)));
	// Define lddw helpers
	SymbolMap lddwSyms;
	std::vector<std::string> definedLddwHelpers;
	const auto tryDefineLddwHelper = [&](const char *name, void *func) {
		if (func) {
			spdlog::debug("Defining LDDW helper {} with addr {:x}",
				      name, (uintptr_t)func);
			auto sym =
				JITEvaluatedSymbol::fromPointer(vm->map_by_fd);
			sym.setFlags(JITSymbolFlags::Callable |
				     JITSymbolFlags::Exported);
			lddwSyms.try_emplace(jit->mangleAndIntern(name), sym);
			definedLddwHelpers.push_back(name);
		}
	};
	tryDefineLddwHelper(LDDW_HELPER_MAP_BY_FD, (void *)vm->map_by_fd);
	tryDefineLddwHelper(LDDW_HELPER_MAP_BY_IDX, (void *)vm->map_by_idx);
	tryDefineLddwHelper(LDDW_HELPER_MAP_VAL, (void *)vm->map_val);
	tryDefineLddwHelper(LDDW_HELPER_CODE_ADDR, (void *)vm->code_addr);
	tryDefineLddwHelper(LDDW_HELPER_VAR_ADDR, (void *)vm->var_addr);
	ExitOnErr(mainDylib.define(absoluteSymbols(lddwSyms)));
	auto bpfModule = ExitOnErr(
		generateModule(*jit, extFuncNames, definedLddwHelpers));
	bpfModule.withModuleDo([](auto &M) { optimizeModule(M); });
	ExitOnErr(jit->addIRModule(std::move(bpfModule)));
	auto func = ExitOnErr(jit->lookup("bpf_main"));
	this->jit = std::move(jit);
	return func.toPtr<ebpf_jit_fn>();
}
