// SPDX-License-Identifier: MIT
#include "rocm_ebpf_compiler.hpp"
#include "rocm_device_helpers.inc"
#include "rocm_channel_abi.hpp"
#include <llvmbpf.hpp>
#include <llvm/AsmParser/Parser.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/SourceMgr.h>
#include <stdexcept>

namespace bpftime::attach
{
std::string compile_rocm_ebpf(const std::vector<ebpf_inst> &instructions,
			      const std::string &entry,
			      const std::vector<rocm_map_binding> &maps)
{
	if (instructions.empty() || instructions.size() >= UINT16_MAX)
		throw std::runtime_error("Invalid ROCm eBPF program length");
	std::vector<bool> continuation(instructions.size(), false);
	for (size_t pc = 0; pc < instructions.size(); ++pc) {
		if (instructions[pc].opcode != EBPF_OP_LDDW)
			continue;
		if (++pc >= instructions.size())
			throw std::runtime_error("Incomplete LDDW");
		const auto &high = instructions[pc];
		if (high.opcode || high.dst || high.src || high.offset)
			throw std::runtime_error("Invalid LDDW continuation");
		continuation[pc] = true;
	}
	for (size_t pc = 0; pc < instructions.size(); ++pc) {
		const auto &i = instructions[pc];
		if (i.dst > 10 || i.src > 10)
			throw std::runtime_error("Invalid eBPF register");
		if (i.opcode == EBPF_OP_LDDW) {
			if (i.src == 1) {
				auto map =
					std::find_if(maps.begin(), maps.end(),
						     [&](const auto &m) {
							     return m.fd ==
								    i.imm;
						     });
				if (map == maps.end() || !map->invoke ||
				    map->info.map_type != 2 ||
				    !map->info.enabled ||
				    map->info.key_size != 4 ||
				    map->info.max_entries <= 0 ||
				    map->info.value_size <= 0 ||
				    map->info.value_size >
					    int(rocm_max_value_size) ||
				    instructions[pc + 1].imm != 0)
					throw std::runtime_error(
						"ROCm requires a retained Array map with value size <= 4096");
			} else if (i.src != 0)
				throw std::runtime_error(
					"ROCm global-data relocations are not supported yet");
			if (++pc >= instructions.size())
				throw std::runtime_error("Incomplete LDDW");
		} else if ((i.opcode & 0xf7) == EBPF_OP_CALL) {
			if (i.opcode != EBPF_OP_CALL || i.src != 0 ||
			    (i.imm != rocm_test_println_id &&
			     (i.imm < 1 || i.imm > 3)))
				throw std::runtime_error(
					"ROCm supports map helpers 1/2/3 and bpf_test_println (506)");
		} else if ((i.opcode & 7) == EBPF_CLS_JMP ||
			   (i.opcode & 7) == EBPF_CLS_JMP32) {
			if (i.opcode == EBPF_OP_EXIT)
				continue;
			const int64_t target =
				int64_t(pc) + 1 +
				(i.opcode == EBPF_OP_JA_IMM ? i.imm : i.offset);
			if (target < 0 ||
			    target >= int64_t(instructions.size()) ||
			    continuation[target])
				throw std::runtime_error(
					"Invalid eBPF branch target");
		}
	}
	llvmbpf_vm vm;
	// This subset has no BPF subprograms. Kernel-compatible lifting gives
	// it an exact-sized stack and avoids the userspace call-stack emulator.
	vm.set_kernel_compatible_mode(true);
	for (unsigned id : { 1, 2, 3 })
		vm.register_external_function(id, "rocm_map_helper", nullptr);
	vm.register_external_function(rocm_test_println_id, "bpf_test_println",
				      nullptr);
	if (vm.load_code(instructions.data(),
			 instructions.size() * sizeof(ebpf_inst)))
		throw std::runtime_error(vm.get_error_message());
	auto ir = vm.generate_llvm_ir(entry);
	if (!ir)
		throw std::runtime_error(vm.get_error_message());
	return std::move(*ir);
}

llvm::Expected<std::unique_ptr<llvm::Module>>
create_rocm_module(const rocm_probe_snapshot &probes,
		   llvm::LLVMContext &context, uint64_t channel,
		   const std::map<int, uint64_t> &maps)
{
	auto buffer = llvm::MemoryBufferRef(
		llvm::StringRef(
			reinterpret_cast<const char *>(rocm_device_helpers),
			sizeof(rocm_device_helpers)),
		"bpftime.rocm.helpers");
	auto parsed = llvm::parseBitcodeFile(buffer, context);
	if (!parsed)
		return parsed.takeError();
	auto module = std::move(*parsed);
	for (const auto *name :
	     { "_bpf_helper_ext_0506", "_bpf_helper_ext_0001",
	       "_bpf_helper_ext_0002", "_bpf_helper_ext_0003" }) {
		module->getFunction(name)->setLinkage(
			llvm::GlobalValue::ExternalLinkage);
		module->getFunction(name)->addFnAttr(
			llvm::Attribute::AlwaysInline);
	}
	auto *address = module->getFunction("bpftime_rocm_channel_address");
	auto *body = llvm::BasicBlock::Create(context, "entry", address);
	llvm::IRBuilder<> address_builder(body);
	address_builder.CreateRet(address_builder.getInt64(channel));
	address->addFnAttr(llvm::Attribute::AlwaysInline);
	for (const char *name : { "llvm.compiler.used", "llvm.used" })
		if (auto *used = module->getGlobalVariable(name))
			used->eraseFromParent();
	for (const auto &probe : probes.probes) {
		llvm::SMDiagnostic diagnostic;
		auto bpf = llvm::parseAssemblyString(probe->ir, diagnostic,
						     context);
		if (!bpf)
			return llvm::make_error<llvm::StringError>(
				"Cannot parse eBPF IR",
				llvm::inconvertibleErrorCode());
		bpf->setTargetTriple(module->getTargetTriple());
		bpf->setDataLayout(module->getDataLayout());
		for (auto it = bpf->global_begin(); it != bpf->global_end();) {
			auto &global = *it++;
			auto name = global.getName();
			if (!name.consume_front("__llvmbpf_pseudo_map_fd_"))
				continue;
			uint32_t fd;
			if (name.getAsInteger(16, fd) || !maps.contains(fd) ||
			    !channel)
				return llvm::make_error<llvm::StringError>(
					"Unbound ROCm map relocation",
					llvm::inconvertibleErrorCode());
			global.replaceAllUsesWith(
				llvm::ConstantExpr::getIntToPtr(
					llvm::ConstantInt::get(
						llvm::Type::getInt64Ty(context),
						maps.at(fd)),
					global.getType()));
			global.eraseFromParent();
		}
		// AMDGPU allocas use AS5. Convert private pointers to flat
		// before representing their values in the eBPF ABI's 64-bit
		// registers.
		for (auto &f : *bpf)
			for (auto &bb : f)
				for (auto it = bb.begin(); it != bb.end();) {
					auto *a =
						llvm::dyn_cast<llvm::AllocaInst>(
							&*it++);
					if (!a || a->getAddressSpace() != 0)
						continue;
					llvm::IRBuilder<> builder(a);
					auto *local = builder.CreateAlloca(
						a->getAllocatedType(), 5,
						a->getArraySize(),
						a->getName() + ".private");
					local->setAlignment(a->getAlign());
					a->replaceAllUsesWith(
						builder.CreateAddrSpaceCast(
							local, a->getType()));
					a->eraseFromParent();
				}
		if (llvm::Linker::linkModules(*module, std::move(bpf)))
			return llvm::make_error<llvm::StringError>(
				"Cannot link eBPF IR",
				llvm::inconvertibleErrorCode());
	}
	std::string diagnostics;
	llvm::raw_string_ostream out(diagnostics);
	if (llvm::verifyModule(*module, &out))
		return llvm::make_error<llvm::StringError>(
			diagnostics, llvm::inconvertibleErrorCode());
	return module;
}
} // namespace bpftime::attach
