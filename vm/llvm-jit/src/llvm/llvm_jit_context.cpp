/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */

#ifdef WIN32
#pragma warning(disable : 4141 4244 4291 4146 4267 4275 4624 4800)
#endif

#include <string.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LegacyPassNameParser.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Config/llvm-config.h>
#if LLVM_VERSION_MAJOR >= 16
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/IPO.h>
#if LLVM_VERSION_MAJOR < 17
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#endif
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>

#if LLVM_VERSION_MAJOR >= 17
#include <llvm/ExecutionEngine/MCJIT.h>
#include <typeinfo>
#include <llvm-c/ExecutionEngine.h>
#include "llvm/LTO/LTOBackend.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/LTO/LTO.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ModuleSymbolTable.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/Transforms/IPO/WholeProgramDevirt.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/LoopAccessAnalysisPrinter.h"
#include "llvm/Transforms/Scalar/LoopDataPrefetch.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopDistribute.h"
#include "llvm/Transforms/Scalar/LoopFuse.h"
#include "llvm/Transforms/Scalar/LoopIdiomRecognize.h"
#include "llvm/Transforms/Scalar/LoopInstSimplify.h"
#include "llvm/Transforms/Scalar/LoopLoadElimination.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopPredication.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopSimplifyCFG.h"
#include "llvm/Transforms/Scalar/LoopSink.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/LoopUnrollAndJamPass.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#endif

// Disappears in LLVM 15
#if LLVM_VERSION_MAJOR >= 14
#include <llvm/MC/TargetRegistry.h>
#else
#include <llvm/Support/TargetRegistry.h>
#endif

#if LLVM_VERSION_MAJOR >= 10
#include <llvm/InitializePasses.h>
#include <llvm/Support/CodeGen.h>
#endif

#include "llvm_jit_context.hpp"
#include "bpftime_vm_compat.hpp"
#include "compiler_utils.hpp"
#include "spdlog/spdlog.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>

#include "llvm/IR/Module.h"
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Error.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Support/Host.h>
#include <llvm/MC/TargetRegistry.h>
#include <memory>
#include <pthread.h>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <system_error>
#include <utility>
#include <iostream>
#include <string>
#include <spdlog/spdlog.h>
#include <tuple>
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <compat_llvm.hpp>
using namespace llvm;
using namespace llvm::orc;
using namespace bpftime;
using namespace std;

struct spin_lock_guard {
	pthread_spinlock_t *spin;
	spin_lock_guard(pthread_spinlock_t *spin) : spin(spin)
	{
		pthread_spin_lock(spin);
	}
	~spin_lock_guard()
	{
		pthread_spin_unlock(spin);
	}
};

static ExitOnError ExitOnErr;

static void optimizeModule(llvm::Module &M)
{
	// std::cout << "LLVM_VERSION_MAJOR: " << LLVM_VERSION_MAJOR <<
	// std::endl;
#if LLVM_VERSION_MAJOR >= 17
	// =====================
	// Create the analysis managers.
	// These must be declared in this order so that they are destroyed in
	// the correct order due to inter-analysis-manager references.
	LoopAnalysisManager LAM;
	FunctionAnalysisManager FAM;
	CGSCCAnalysisManager CGAM;
	ModuleAnalysisManager MAM;

	// Create the new pass manager builder.
	// Take a look at the PassBuilder constructor parameters for more
	// customization, e.g. specifying a TargetMachine or various debugging
	// options.
	PassBuilder PB;

	// Register all the basic analyses with the managers.
	PB.registerModuleAnalyses(MAM);
	PB.registerCGSCCAnalyses(CGAM);
	PB.registerFunctionAnalyses(FAM);
	PB.registerLoopAnalyses(LAM);
	PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

	// Create the pass manager.
	// This one corresponds to a typical -O2 optimization pipeline.
	ModulePassManager MPM =
		PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);

	// Optimize the IR!
	MPM.run(M, MAM);
	// =====================================
#else
	llvm::legacy::PassManager PM;

	llvm::PassManagerBuilder PMB;
	PMB.OptLevel = 3;
	PMB.populateModulePassManager(PM);

	PM.run(M);
#endif
}

#if defined(__arm__) || defined(_M_ARM)
extern "C" void __aeabi_unwind_cpp_pr1();
#endif

static int llvm_initialized = 0;

llvm_bpf_jit_context::llvm_bpf_jit_context(
	class bpftime::vm::llvm::bpftime_llvm_jit_vm *vm)
	: vm(vm)
{
	using namespace llvm;
	int zero = 0;
	if (__atomic_compare_exchange_n(&llvm_initialized, &zero, 1, false,
					__ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
		SPDLOG_INFO("Initializing llvm");
		InitializeNativeTarget();
		InitializeNativeTargetAsmPrinter();
	}
	compiling = std::make_unique<pthread_spinlock_t>();
	pthread_spin_init(compiling.get(), PTHREAD_PROCESS_PRIVATE);
}
void llvm_bpf_jit_context::do_jit_compile()
{
	auto [jit, extFuncNames, definedLddwHelpers] =
		create_and_initialize_lljit_instance();
	auto bpfModule = ExitOnErr(
		generateModule(extFuncNames, definedLddwHelpers, true));
	bpfModule.withModuleDo([](auto &M) { optimizeModule(M); });
	ExitOnErr(jit->addIRModule(std::move(bpfModule)));
	this->jit = std::move(jit);
}

bpftime::vm::compat::precompiled_ebpf_function llvm_bpf_jit_context::compile()
{
	spin_lock_guard guard(compiling.get());
	if (!this->jit.has_value()) {
		do_jit_compile();
	} else {
		SPDLOG_DEBUG("LLVM-JIT: already compiled");
	}

	return this->get_entry_address();
}

llvm_bpf_jit_context::~llvm_bpf_jit_context()
{
	pthread_spin_destroy(compiling.get());
}

std::vector<uint8_t> llvm_bpf_jit_context::do_aot_compile(
	const std::vector<std::string> &extFuncNames,
	const std::vector<std::string> &lddwHelpers, bool print_ir)
{
	SPDLOG_DEBUG("AOT: start");
	if (auto module = generateModule(extFuncNames, lddwHelpers, false);
	    module) {
		auto defaultTargetTriple = llvm::sys::getDefaultTargetTriple();
		SPDLOG_DEBUG("AOT: target triple: {}", defaultTargetTriple);
		return module->withModuleDo([&](auto &module)
						    -> std::vector<uint8_t> {
			if (print_ir) {
				module.print(errs(), nullptr);
			}
			optimizeModule(module);
			module.setTargetTriple(defaultTargetTriple);
			std::string error;
			auto target = TargetRegistry::lookupTarget(
				defaultTargetTriple, error);
			if (!target) {
				SPDLOG_ERROR(
					"AOT: Failed to get local target: {}",
					error);
				throw std::runtime_error(
					"Unable to get local target");
			}
			auto targetMachine = target->createTargetMachine(
				defaultTargetTriple, "generic", "",
				TargetOptions(), Reloc::PIC_);
			if (!targetMachine) {
				SPDLOG_ERROR("Unable to create target machine");
				throw std::runtime_error(
					"Unable to create target machine");
			}
			module.setDataLayout(targetMachine->createDataLayout());
			SmallVector<char, 0> objStream;
			std::unique_ptr<raw_svector_ostream> BOS =
				std::make_unique<raw_svector_ostream>(
					objStream);

			legacy::PassManager pass;
// auto FileType = CGFT_ObjectFile;
#if LLVM_VERSION_MAJOR >= 18
			if (targetMachine->addPassesToEmitFile(
				    pass, *BOS, nullptr,
				    CodeGenFileType::ObjectFile)) {
#elif LLVM_VERSION_MAJOR >= 10
			if (targetMachine->addPassesToEmitFile(
				    pass, *BOS, nullptr, CGFT_ObjectFile)) {
#elif LLVM_VERSION_MAJOR >= 8
			if (targetMachine->addPassesToEmitFile(
				    pass, *BOS, nullptr,
				    TargetMachine::CGFT_ObjectFile)) {
#else
			if (targetMachine->addPassesToEmitFile(
				    pass, *BOS, TargetMachine::CGFT_ObjectFile,
				    true)) {
#endif
				SPDLOG_ERROR(
					"Unable to emit module for target machine");
				throw std::runtime_error(
					"Unable to emit module for target machine");
			}

			pass.run(module);
			SPDLOG_INFO("AOT: done, received {} bytes",
				    objStream.size());

			std::vector<uint8_t> result(objStream.begin(),
						    objStream.end());
			return result;
		});
	} else {
		std::string buf;
		raw_string_ostream os(buf);
		os << module.takeError();
		SPDLOG_ERROR("Unable to generate module: {}", buf);
		throw std::runtime_error("Unable to generate llvm module");
	}
}

std::vector<uint8_t> llvm_bpf_jit_context::do_aot_compile(bool print_ir)
{
	std::vector<std::string> extNames, lddwNames;
	for (uint32_t i = 0; i < std::size(vm->ext_funcs); i++) {
		if (vm->ext_funcs[i].has_value()) {
#if LLVM_VERSION_MAJOR >= 16
			extNames.emplace_back(ext_func_sym(i));
#else
			extNames.push_back(ext_func_sym(i));
#endif
		}
	}

	const auto tryDefineLddwHelper = [&](const char *name, void *func) {
		if (func) {
#if LLVM_VERSION_MAJOR >= 16
			lddwNames.emplace_back(name);
#else
			lddwNames.push_back(name);
#endif
		}
	};
	// Only map_val will have a chance to be called at runtime
	tryDefineLddwHelper(LDDW_HELPER_MAP_VAL, (void *)vm->map_val);
	// These symbols won't be used at runtime
	// tryDefineLddwHelper(LDDW_HELPER_MAP_BY_FD, (void *)vm->map_by_fd);
	// tryDefineLddwHelper(LDDW_HELPER_MAP_BY_IDX, (void *)vm->map_by_idx);
	// tryDefineLddwHelper(LDDW_HELPER_CODE_ADDR, (void *)vm->code_addr);
	// tryDefineLddwHelper(LDDW_HELPER_VAR_ADDR, (void *)vm->var_addr);
	return this->do_aot_compile(extNames, lddwNames, print_ir);
}

void llvm_bpf_jit_context::load_aot_object(const std::vector<uint8_t> &buf)
{
	SPDLOG_INFO("LLVM-JIT: Loading aot object");
	if (jit.has_value()) {
		SPDLOG_ERROR("Unable to load aot object: already compiled");
		throw std::runtime_error(
			"Unable to load aot object: already compiled");
	}
	auto buffer = MemoryBuffer::getMemBuffer(
		StringRef((const char *)buf.data(), buf.size()));
	auto [jit, extFuncNames, definedLddwHelpers] =
		create_and_initialize_lljit_instance();
	if (auto err = jit->addObjectFile(std::move(buffer)); err) {
		std::string buf;
		raw_string_ostream os(buf);
		os << err;
		SPDLOG_CRITICAL("Unable to add object file: {}", buf);
		throw std::runtime_error("Failed to load AOT object");
	}
	this->jit = std::move(jit);
	// Test getting entry function
	this->get_entry_address();
}

std::tuple<std::unique_ptr<llvm::orc::LLJIT>, std::vector<std::string>,
	   std::vector<std::string> >
llvm_bpf_jit_context::create_and_initialize_lljit_instance()
{
	// Create a JIT builder
	SPDLOG_DEBUG("LLVM-JIT: Creating LLJIT instance");
	auto jit = ExitOnErr(LLJITBuilder().create());

	auto &mainDylib = jit->getMainJITDylib();
	std::vector<std::string> extFuncNames;
	// insert the helper functions
	SymbolMap extSymbols;
	for (uint32_t i = 0; i < std::size(vm->ext_funcs); i++) {
		if (vm->ext_funcs[i].has_value()) {
			auto sym = JITEvaluatedSymbol::fromPointer(
				vm->ext_funcs[i]->fn);
			auto symName = jit->getExecutionSession().intern(
				ext_func_sym(i));
			sym.setFlags(JITSymbolFlags::Callable |
				     JITSymbolFlags::Exported);

#if LLVM_VERSION_MAJOR < 17
			extSymbols.try_emplace(symName, sym);
			extFuncNames.push_back(ext_func_sym(i));
#else
			auto symbol = ::llvm::orc::ExecutorSymbolDef(
				::llvm::orc::ExecutorAddr(sym.getAddress()),
				sym.getFlags());
			extSymbols.try_emplace(symName, symbol);
			extFuncNames.emplace_back(ext_func_sym(i));
#endif
		}
	}
#if defined(__arm__) || defined(_M_ARM)
	SPDLOG_INFO("Defining __aeabi_unwind_cpp_pr1 on arm32");
	extSymbols.try_emplace(
		jit->getExecutionSession().intern("__aeabi_unwind_cpp_pr1"),
		JITEvaluatedSymbol::fromPointer(__aeabi_unwind_cpp_pr1));
#endif
	ExitOnErr(mainDylib.define(absoluteSymbols(extSymbols)));
	// Define lddw helpers
	SymbolMap lddwSyms;
	std::vector<std::string> definedLddwHelpers;
	const auto tryDefineLddwHelper = [&](const char *name, void *func) {
		if (func) {
			SPDLOG_DEBUG("Defining LDDW helper {} with addr {:x}",
				     name, (uintptr_t)func);
			auto sym = JITEvaluatedSymbol::fromPointer(func);
			// printf("The type of sym %s\n", typeid(sym).name());
			sym.setFlags(JITSymbolFlags::Callable |
				     JITSymbolFlags::Exported);

#if LLVM_VERSION_MAJOR < 17
			lddwSyms.try_emplace(
				jit->getExecutionSession().intern(name), sym);
			definedLddwHelpers.push_back(name);
#else
			auto symbol = ::llvm::orc::ExecutorSymbolDef(
				::llvm::orc::ExecutorAddr(sym.getAddress()),
				sym.getFlags());
			lddwSyms.try_emplace(
				jit->getExecutionSession().intern(name),
				symbol);
			definedLddwHelpers.emplace_back(name);
#endif
		}
	};
	// Only map_val will have a chance to be called at runtime, so it's the
	// only symbol to be defined
	tryDefineLddwHelper(LDDW_HELPER_MAP_VAL, (void *)vm->map_val);
	// These symbols won't be used at runtime
	// tryDefineLddwHelper(LDDW_HELPER_MAP_BY_FD, (void *)vm->map_by_fd);
	// tryDefineLddwHelper(LDDW_HELPER_MAP_BY_IDX, (void *)vm->map_by_idx);
	// tryDefineLddwHelper(LDDW_HELPER_CODE_ADDR, (void *)vm->code_addr);
	// tryDefineLddwHelper(LDDW_HELPER_VAR_ADDR, (void *)vm->var_addr);
	ExitOnErr(mainDylib.define(absoluteSymbols(lddwSyms)));
	return { std::move(jit), extFuncNames, definedLddwHelpers };
}

bpftime::vm::compat::precompiled_ebpf_function
llvm_bpf_jit_context::get_entry_address()
{
	if (!this->jit.has_value()) {
		SPDLOG_CRITICAL(
			"Not compiled yet. Unable to get entry func address");
		throw std::runtime_error("Not compiled yet");
	}
	if (auto err = (*jit)->lookup("bpf_main"); !err) {
		std::string buf;
		raw_string_ostream os(buf);
		os << err.takeError();
		SPDLOG_CRITICAL("Unable to find symbol `bpf_main`: {}", buf);
		throw std::runtime_error("Unable to link symbol `bpf_main`");
	} else {
		auto addr = err->toPtr<vm::compat::precompiled_ebpf_function>();
		SPDLOG_DEBUG("LLVM-JIT: Entry func is {:x}", (uintptr_t)addr);
		return addr;
	}
}
