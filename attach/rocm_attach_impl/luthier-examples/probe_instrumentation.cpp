// SPDX-License-Identifier: MIT
// Uses Luthier's public CRTP traits, as composed by Tool.h, without the HIP
// offload-parser trait: this example supplies its own LLVM instrumentation IR.
#include <luthier/Common/Singleton.h>
#include <luthier/HSA/ExecutableSymbol.h>
#include <luthier/Rocprofiler/ApiTableSnapshot.h>
#include <luthier/Rocprofiler/RocprofilerSdkLibrary.h>
#include <luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h>
#include <luthier/ToolCodeGen/PrototypeCallGraph.h>
#include <luthier/Tooling/InstrumentationPipelineTrait.h>
#include <luthier/Tooling/InstrumentedKernelLoaderAndLauncher.h>
#include <luthier/Tooling/LoadedCodeObjectCache.h>
#include <luthier/Tooling/PacketMonitorTrait.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/TargetSelect.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>

namespace
{
constexpr unsigned Weight = 7;
constexpr const char *CounterName = "bpftime_luthier_counts";
enum class Mode { Roundtrip, Entry, EntryExit };

struct Runtime {
	FILE *report = nullptr;
	Mode mode = Mode::Entry;
	bool initialized = false;
	std::unique_ptr<luthier::DynamicLibrary> library;
	std::unique_ptr<luthier::rocprofiler::HsaApiTableSnapshot<CoreApiTable>>
		core;
	std::unique_ptr<luthier::rocprofiler::HsaApiTableSnapshot<AmdExtTable>>
		amd;
	std::unique_ptr<luthier::rocprofiler::HsaExtensionTableSnapshot<
		HSA_EXTENSION_AMD_LOADER>>
		loader;
};

Runtime &runtime()
{
	// ROCprofiler may configure the tool before this DSO's global
	// constructors.
	static auto *state = new Runtime;
	return *state;
}

void report(std::initializer_list<llvm::json::Object::KV> object)
{
	if (!runtime().report)
		return;
	std::string text;
	llvm::raw_string_ostream os(text);
	os << llvm::json::Value(llvm::json::Object(object)) << '\n';
	std::fwrite(text.data(), 1, text.size(), runtime().report);
	std::fflush(runtime().report);
}

void reportError(llvm::Error error)
{
	report({ { "status", "error" },
		 { "message", llvm::toString(std::move(error)) } });
}

struct ProbeTool : luthier::Singleton<ProbeTool>,
		   luthier::LoadedCodeObjectCacheTrait<ProbeTool>,
		   luthier::InstrumentedKernelLoaderAndLauncherTrait<ProbeTool>,
		   luthier::InstrumentationPipelineTrait<ProbeTool,
							 llvm::MachineFunction>,
		   luthier::PacketMonitorTrait<ProbeTool> {
	ProbeTool(CreationKey, const luthier::DynamicLibrary &lib,
		  const luthier::rocprofiler::HsaApiTableSnapshot<CoreApiTable>
			  &core,
		  const luthier::rocprofiler::HsaApiTableSnapshot<AmdExtTable>
			  &amd,
		  const luthier::rocprofiler::HsaExtensionTableSnapshot<
			  HSA_EXTENSION_AMD_LOADER> &loader,
		  llvm::Error &error)
		: luthier::Singleton<ProbeTool>(),
		  luthier::LoadedCodeObjectCacheTrait<ProbeTool>(lib, core,
								 loader, error),
		  luthier::InstrumentedKernelLoaderAndLauncherTrait<ProbeTool>(
			  lib, core, amd, loader, error),
		  luthier::InstrumentationPipelineTrait<ProbeTool,
							llvm::MachineFunction>(
			  core, amd, loader),
		  luthier::PacketMonitorTrait<ProbeTool>(lib, core, amd, loader,
							 error)
	{
	}

	std::recursive_mutex mutex;
	std::string compilationError;

	std::unique_ptr<llvm::Module>
	createInstrumentationModule(llvm::LLVMContext &context)
	{
		auto module = std::make_unique<llvm::Module>("bpftime.probe",
							     context);
		llvm::IRBuilder<> builder(context);
		auto *arrayType = llvm::ArrayType::get(builder.getInt64Ty(), 2);
		auto *counts = new llvm::GlobalVariable(
			*module, arrayType, false,
			llvm::GlobalValue::ExternalLinkage,
			llvm::ConstantAggregateZero::get(arrayType),
			CounterName, nullptr, llvm::GlobalValue::NotThreadLocal,
			1);
		counts->setAlignment(llvm::Align(8));
		auto *functionType = llvm::FunctionType::get(
			builder.getVoidTy(),
			{ builder.getInt32Ty(), builder.getInt64Ty() }, false);
		auto *hook = llvm::Function::Create(
			functionType, llvm::GlobalValue::InternalLinkage,
			"bpftime_count_hook", *module);
		hook->setDoesNotThrow();
		hook->setDoesNotRecurse();
		builder.SetInsertPoint(
			llvm::BasicBlock::Create(context, "entry", hook));
		auto *slot = builder.CreateInBoundsGEP(arrayType, counts,
						       { builder.getInt32(0),
							 hook->getArg(0) });
		builder.CreateAtomicRMW(llvm::AtomicRMWInst::Add, slot,
					hook->getArg(1), llvm::Align(8),
					llvm::AtomicOrdering::Monotonic);
		builder.CreateRetVoid();
		llvm::raw_string_ostream errors(compilationError);
		llvm::verifyModule(*module, &errors);
		return module;
	}

	llvm::PreservedAnalyses run(luthier::Prototype &prototype,
				    luthier::PrototypeAnalysisManager &analyses)
	{
		if (runtime().mode == Mode::Roundtrip)
			return llvm::PreservedAnalyses::all();
		auto &module = prototype.getInstrumentationModule();
		auto &functions =
			analyses.getResult<
					luthier::IModuleFunctionAnalysisManagerPrototypeProxy>(
					prototype)
				.getManager();
		auto *hook = module.getFunction("bpftime_count_hook");
		unsigned entrySites = 0, exitSites = 0;
		prototype.forEachTargetMF(analyses, [&](llvm::MachineFunction
								&function) {
			if (function.empty() ||
			    function.getName() != "bpftime_luthier_vector_add")
				return;
			auto insert = [&](llvm::MachineInstr &instruction,
					  unsigned slot) {
				llvm::IRBuilder<> builder(module.getContext());
				auto payload = prototype.createInjectedPayload(
					*hook, instruction, functions,
					{ builder.getInt32(slot),
					  builder.getInt64(Weight) },
					luthier::PayloadLaneMode::SingleLane);
				if (!payload)
					compilationError += llvm::toString(
						payload.takeError());
			};
			// Collect original sites before inserting PATCHPOINT
			// instructions.
			llvm::SmallVector<llvm::MachineInstr *> exits;
			for (auto &block : function)
				for (auto &instruction : block)
					if (function.getSubtarget()
						    .getInstrInfo()
						    ->getName(
							    instruction
								    .getOpcode()) ==
					    "S_ENDPGM")
						exits.push_back(&instruction);
			insert(function.front().front(), 0);
			++entrySites;
			if (runtime().mode == Mode::EntryExit)
				for (auto *instruction : exits) {
					insert(*instruction, 1);
					++exitSites;
				}
		});
		if (entrySites != 1 ||
		    (runtime().mode == Mode::EntryExit && exitSites == 0))
			compilationError +=
				"Expected one kernel entry and a wave termination site";
		llvm::PreservedAnalyses preserved =
			llvm::PreservedAnalyses::none();
		preserved.preserve<
			luthier::TargetModuleAnalysisManagerPrototypeProxy>();
		preserved.preserve<
			luthier::TargetFunctionAnalysisManagerPrototypeProxy>();
		preserved.preserve<
			luthier::TargetMachineFunctionAnalysisManagerPrototypeProxy>();
		preserved.preserve<
			luthier::IModuleAnalysisManagerPrototypeProxy>();
		preserved.preserve<
			luthier::IModuleFunctionAnalysisManagerPrototypeProxy>();
		preserved.preserve<
			luthier::IModuleMachineFunctionAnalysisManagerPrototypeProxy>();
		preserved
			.abandon<luthier::InjectedPayloadAndInstPointAnalysis>();
		preserved.abandon<luthier::PrototypeCallGraphAnalysis>();
		return preserved;
	}

	void onPackets(const hsa_queue_t &queue, uint64_t,
		       llvm::ArrayRef<luthier::hsa::AqlPacket> packets,
		       hsa_amd_queue_intercept_packet_writer writer);
};

void ProbeTool::onPackets(const hsa_queue_t &queue, uint64_t,
			  llvm::ArrayRef<luthier::hsa::AqlPacket> packets,
			  hsa_amd_queue_intercept_packet_writer writer)
{
	std::lock_guard<std::recursive_mutex> guard(mutex);
	for (auto packet : packets) {
		auto *dispatch = packet.asKernelDispatch();
		if (!dispatch) {
			writer(&packet, 1);
			continue;
		}
		const auto original = *dispatch;
		const auto *descriptor = reinterpret_cast<
			const llvm::amdhsa::kernel_descriptor_t *>(
			dispatch->kernel_object);
		auto core = runtime().core->getTable();
		auto amd = runtime().amd->getTable();
		// HIP may dispatch internal copy kernels even for hipMemcpy
		// calls.
		hsa_executable_t executable{};
		hsa_amd_pointer_info_t pointerInfo{};
		pointerInfo.size = sizeof(pointerInfo);
		if (runtime().loader->getTable()
				    .hsa_ven_amd_loader_query_executable(
					    descriptor, &executable) !=
			    HSA_STATUS_SUCCESS ||
		    amd.callFunction<hsa_amd_pointer_info>(
			    const_cast<void *>(
				    static_cast<const void *>(descriptor)),
			    &pointerInfo, nullptr, nullptr,
			    nullptr) != HSA_STATUS_SUCCESS) {
			report({ { "status", "error" },
				 { "message",
				   "Cannot identify dispatch executable/agent" } });
			writer(&original, 1);
			continue;
		}
		hsa_executable_symbol_t targetSymbol{};
		uint64_t targetObject = 0;
		for (const char *name : { "bpftime_luthier_vector_add.kd",
					  "bpftime_luthier_vector_add" }) {
			if (core.callFunction<hsa_executable_get_symbol_by_name>(
				    executable, name, &pointerInfo.agentOwner,
				    &targetSymbol) == HSA_STATUS_SUCCESS) {
				(void)core.callFunction<
					hsa_executable_symbol_get_info>(
					targetSymbol,
					HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
					&targetObject);
				break;
			}
		}
		if (targetObject != dispatch->kernel_object) {
			writer(&original, 1);
			continue;
		}
		auto variant = overrideWithInstrumented(*dispatch, queue, 0);
		if (!variant) {
			llvm::consumeError(variant.takeError());
			compilationError.clear();
			auto object = runInstrumentationPipelineForDispatch(
				*descriptor);
			if (!object || !compilationError.empty()) {
				if (!object)
					reportError(object.takeError());
				else
					report({ { "status", "error" },
						 { "message",
						   compilationError } });
				writer(&original, 1);
				continue;
			}
			auto loaded = loadInstrumented(std::move(*object),
						       descriptor, 0);
			if (!loaded) {
				reportError(loaded.takeError());
				writer(&original, 1);
				continue;
			}
			variant = overrideWithInstrumented(*dispatch, queue, 0);
		}
		if (!variant) {
			reportError(variant.takeError());
			writer(&original, 1);
			continue;
		}
		auto symbol = lookupGlobalVariable(CounterName, descriptor);
		if (!symbol) {
			reportError(symbol.takeError());
			writer(&original, 1);
			continue;
		}
		auto address =
			luthier::hsa::executableSymbolGetAddress(core, *symbol);
		uint64_t counts[2] = { 0, 0 };
		if (!address) {
			reportError(address.takeError());
			writer(&original, 1);
			continue;
		}
		void *deviceCounts = reinterpret_cast<void *>(*address);
		if (core.callFunction<hsa_memory_copy>(deviceCounts, counts,
						       sizeof(counts)) !=
		    HSA_STATUS_SUCCESS) {
			report({ { "status", "error" },
				 { "message",
				   "Cannot clear device counters" } });
			writer(&original, 1);
			continue;
		}
		const bool ownsSignal = dispatch->completion_signal.handle == 0;
		if (ownsSignal &&
		    amd.callFunction<&AmdExtTable::hsa_amd_signal_create_fn>(
			    1, 0, nullptr, 0, &dispatch->completion_signal) !=
			    HSA_STATUS_SUCCESS) {
			report({ { "status", "error" },
				 { "message",
				   "Cannot create completion signal" } });
			writer(&original, 1);
			continue;
		}
		const auto before = core.callFunction<
			&CoreApiTable::hsa_signal_load_scacquire_fn>(
			dispatch->completion_signal);
		uint64_t frequency = 0;
		core.callFunction<hsa_system_get_info>(
			HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency);
		writer(dispatch, 1);
		const auto after = core.callFunction<
			&CoreApiTable::hsa_signal_wait_scacquire_fn>(
			dispatch->completion_signal, HSA_SIGNAL_CONDITION_LT,
			before, frequency * 30, HSA_WAIT_STATE_BLOCKED);
		if (after >= before) {
			// Never replay an already submitted kernel or free live
			// kernargs.
			new luthier::ExtendedKernargBuffer(std::move(*variant));
			report({ { "status", "error" },
				 { "message",
				   "Dispatch timeout; retaining kernargs" } });
			continue;
		}
		if (ownsSignal)
			core.callFunction<&CoreApiTable::hsa_signal_destroy_fn>(
				dispatch->completion_signal);
		if (core.callFunction<hsa_memory_copy>(counts, deviceCounts,
						       sizeof(counts)) !=
		    HSA_STATUS_SUCCESS) {
			report({ { "status", "error" },
				 { "message",
				   "Cannot read device counters" } });
			continue;
		}
		report({ { "status", "dispatch" },
			 { "entry", counts[0] },
			 { "exit", counts[1] },
			 { "grid_x", original.grid_size_x },
			 { "block_x", original.workgroup_size_x },
			 { "weight", Weight } });
	}
}

bool initialize()
{
	auto &state = runtime();
	const char *path = std::getenv("BPFTIME_LUTHIER_REPORT");
	const char *mode = std::getenv("BPFTIME_LUTHIER_MODE");
	if (!path || !mode)
		return false;
	state.report = std::fopen(path, "w");
	if (!state.report)
		return false;
	if (std::string(mode) == "roundtrip")
		state.mode = Mode::Roundtrip;
	else if (std::string(mode) == "entry-exit")
		state.mode = Mode::EntryExit;
	else if (std::string(mode) != "entry") {
		report({ { "status", "error" },
			 { "message", "Invalid mode" } });
		return false;
	}
	LLVMInitializeAMDGPUTargetInfo();
	LLVMInitializeAMDGPUTarget();
	LLVMInitializeAMDGPUTargetMC();
	LLVMInitializeAMDGPUAsmPrinter();
	LLVMInitializeAMDGPUAsmParser();
	LLVMInitializeAMDGPUDisassembler();
	LLVMInitializeAMDGPUTargetMCA();
	auto library = luthier::rocprofiler::openRocprofilerSdkLibrary();
	if (!library) {
		reportError(library.takeError());
		return false;
	}
	state.library =
		std::make_unique<luthier::DynamicLibrary>(std::move(*library));
	llvm::Error error = llvm::Error::success();
	state.core = std::make_unique<
		luthier::rocprofiler::HsaApiTableSnapshot<CoreApiTable>>(
		*state.library, error);
	if (error) {
		reportError(std::move(error));
		return false;
	}
	state.amd = std::make_unique<
		luthier::rocprofiler::HsaApiTableSnapshot<AmdExtTable>>(
		*state.library, error);
	if (error) {
		reportError(std::move(error));
		return false;
	}
	state.loader =
		std::make_unique<luthier::rocprofiler::HsaExtensionTableSnapshot<
			HSA_EXTENSION_AMD_LOADER>>(*state.library, error);
	if (error) {
		reportError(std::move(error));
		return false;
	}
	ProbeTool::createInstance(*state.library, *state.core, *state.amd,
				  *state.loader, error);
	state.initialized = true;
	if (error) {
		reportError(std::move(error));
		return false;
	}
	report({ { "status", "initialized" }, { "mode", mode } });
	return true;
}

void finalize(void *)
{
	auto &state = runtime();
	if (state.initialized) {
		ProbeTool::destroyInstance();
		state.initialized = false;
		report({ { "status", "finalized" } });
	}
	if (state.report) {
		std::fclose(state.report);
		state.report = nullptr;
	}
	// Snapshots stay alive: upstream's forwarding wrappers can outlive the
	// tool.
}
} // namespace

extern "C" __attribute__((visibility("default")))
rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t, const char *, uint32_t,
		      rocprofiler_client_id_t *client)
{
	try {
		if (!initialize()) {
			finalize(nullptr);
			return nullptr;
		}
		client->name = "bpftime Luthier examples";
		static rocprofiler_tool_configure_result_t config{
			sizeof(config), nullptr, finalize, nullptr
		};
		return &config;
	} catch (const std::exception &error) {
		report({ { "status", "error" }, { "message", error.what() } });
		finalize(nullptr);
		return nullptr;
	}
}
