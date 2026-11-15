// SPDX-License-Identifier: MIT
#include "rocm_attach_impl.hpp"
#include "rocm_attach_private_data.hpp"
#include "rocm_ebpf_compiler.hpp"
#include <llvm/IR/Verifier.h>
#include <llvm/IR/Instructions.h>
#include <spdlog/spdlog.h>
#include <cstdio>
#include <cerrno>

#define CHECK(x)                                                               \
	do {                                                                   \
		if (!(x)) {                                                    \
			std::fprintf(stderr, "check failed at %d: %s\n",       \
				     __LINE__, #x);                            \
			return 1;                                              \
		}                                                              \
	} while (false)
using namespace bpftime::attach;
int main()
{
	spdlog::set_level(spdlog::level::off);
	CHECK(!has_rocm_probes());
	rocm_attach_impl attach;
	rocm_attach_private_data data;
	data.func_name = "kernel";
	data.instructions = { { EBPF_OP_STW, 10, 0, -8, 0x6968 },
			      { EBPF_OP_MOV64_REG, 1, 10, 0, 0 },
			      { EBPF_OP_ADD64_IMM, 1, 0, 0, -8 },
			      { EBPF_OP_CALL, 0, 0, 0, 506 },
			      { EBPF_OP_MOV64_IMM, 0, 0, 0, 0 },
			      { EBPF_OP_EXIT, 0, 0, 0, 0 } };
	const int entry = attach.create_attach_with_ebpf_callback(
		{}, data, ATTACH_ROCM_PROBE_AND_RETPROBE);
	CHECK(entry > 0);
	data.is_ret_probe = true;
	const int exit = attach.create_attach_with_ebpf_callback(
		{}, data, ATTACH_ROCM_PROBE_AND_RETPROBE);
	CHECK(exit > entry);
	CHECK(has_rocm_probes());
	auto snapshot = snapshot_rocm_probes();
	CHECK(snapshot.probes.size() == 2);
	CHECK(!snapshot.probes[0]->at_return && snapshot.probes[1]->at_return);
	llvm::LLVMContext context;
	auto module = create_rocm_module(snapshot, context);
	if (!module) {
		std::fprintf(stderr, "%s\n",
			     llvm::toString(module.takeError()).c_str());
		return 1;
	}
	CHECK(!llvm::verifyModule(**module));
	auto *function = (*module)->getFunction(snapshot.probes[0]->entry);
	CHECK(function && !function->isDeclaration());
	CHECK(!(*module)->getFunction("_bpf_helper_ext_0506")->isDeclaration());
	unsigned allocas = 0;
	for (auto &block : *function)
		for (auto &inst : block)
			if (auto *a = llvm::dyn_cast<llvm::AllocaInst>(&inst)) {
				CHECK(a->getAddressSpace() == 5);
				++allocas;
			}
	CHECK(allocas > 0);
	data.instructions[3].imm = 14;
	CHECK(attach.create_attach_with_ebpf_callback(
		      {}, data, ATTACH_ROCM_PROBE_AND_RETPROBE) == -EINVAL);
	CHECK(snapshot_rocm_probes().generation == snapshot.generation);
	// Reject map relocations, malformed wide immediates and jumps into
	// their continuation before handing bytecode to LLVM.
	for (const std::vector<ebpf_inst> &bad :
	     { std::vector<ebpf_inst>{ { EBPF_OP_LDDW, 1, 1, 0, 0 },
				       {},
				       { EBPF_OP_EXIT, 0, 0, 0, 0 } },
	       std::vector<ebpf_inst>{ { EBPF_OP_LDDW, 1, 0, 0, 0 } },
	       std::vector<ebpf_inst>{ { EBPF_OP_JA, 0, 0, 1, 0 },
				       { EBPF_OP_LDDW, 1, 0, 0, 0 },
				       {},
				       { EBPF_OP_EXIT, 0, 0, 0, 0 } },
	       std::vector<ebpf_inst>{ { EBPF_OP_CALL, 0, 1, 0, 506 },
				       { EBPF_OP_EXIT, 0, 0, 0, 0 } } }) {
		data.instructions = bad;
		CHECK(attach.create_attach_with_ebpf_callback(
			      {}, data, ATTACH_ROCM_PROBE_AND_RETPROBE) ==
		      -EINVAL);
	}
	CHECK(snapshot_rocm_probes().generation == snapshot.generation);
	CHECK(attach.detach_by_id(entry) == 0);
	CHECK(attach.detach_by_id(entry) == -ENOENT);
	CHECK(snapshot_rocm_probes().generation > snapshot.generation);
	CHECK(snapshot_rocm_probes().probes.size() == 1);
	CHECK(attach.detach_by_id(exit) == 0);
	CHECK(!has_rocm_probes());
	// Snapshots taken for an in-flight dispatch remain valid after detach.
	CHECK(snapshot.probes.size() == 2 && !snapshot.probes[0]->ir.empty());
	return 0;
}
