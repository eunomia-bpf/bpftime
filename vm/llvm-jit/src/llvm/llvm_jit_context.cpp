/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
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
	llvm::legacy::PassManager PM;

	llvm::PassManagerBuilder PMB;
	PMB.OptLevel = 3;
	PMB.populateModulePassManager(PM);

	PM.run(M);
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
	auto bpfModule =
		ExitOnErr(generateModule(extFuncNames, definedLddwHelpers));
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
	if (auto module = generateModule(extFuncNames, lddwHelpers); module) {
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
			if (targetMachine->addPassesToEmitFile(
				    pass, *BOS, nullptr, CGFT_ObjectFile)) {
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
			extNames.push_back(ext_func_sym(i));
		}
	}

	const auto tryDefineLddwHelper = [&](const char *name, void *func) {
		if (func) {
			lddwNames.push_back(name);
		}
	};
	tryDefineLddwHelper(LDDW_HELPER_MAP_BY_FD, (void *)vm->map_by_fd);
	tryDefineLddwHelper(LDDW_HELPER_MAP_BY_IDX, (void *)vm->map_by_idx);
	tryDefineLddwHelper(LDDW_HELPER_MAP_VAL, (void *)vm->map_val);
	tryDefineLddwHelper(LDDW_HELPER_CODE_ADDR, (void *)vm->code_addr);
	tryDefineLddwHelper(LDDW_HELPER_VAR_ADDR, (void *)vm->var_addr);
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
			extSymbols.try_emplace(symName, sym);
			extFuncNames.push_back(ext_func_sym(i));
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
			sym.setFlags(JITSymbolFlags::Callable |
				     JITSymbolFlags::Exported);
			lddwSyms.try_emplace(
				jit->getExecutionSession().intern(name), sym);
			definedLddwHelpers.push_back(name);
		}
	};
	tryDefineLddwHelper(LDDW_HELPER_MAP_BY_FD, (void *)vm->map_by_fd);
	tryDefineLddwHelper(LDDW_HELPER_MAP_BY_IDX, (void *)vm->map_by_idx);
	tryDefineLddwHelper(LDDW_HELPER_MAP_VAL, (void *)vm->map_val);
	tryDefineLddwHelper(LDDW_HELPER_CODE_ADDR, (void *)vm->code_addr);
	tryDefineLddwHelper(LDDW_HELPER_VAR_ADDR, (void *)vm->var_addr);
	ExitOnErr(mainDylib.define(absoluteSymbols(lddwSyms)));
	return { std::move(jit), extFuncNames, definedLddwHelpers };
}

bpftime::vm::compat::precompiled_ebpf_function llvm_bpf_jit_context::get_entry_address()
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
