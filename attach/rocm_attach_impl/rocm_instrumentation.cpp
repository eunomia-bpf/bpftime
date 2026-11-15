// SPDX-License-Identifier: MIT
#include "rocm_instrumentation.hpp"
#include "rocm_ebpf_compiler.hpp"
#include <luthier/Common/Singleton.h>
#include <luthier/HSA/AqlPacket.h>
#include <luthier/Tooling/LoadedCodeObjectCache.h>
#include <luthier/Tooling/InstrumentedKernelLoaderAndLauncher.h>
#include <luthier/Tooling/InstrumentationPipelineTrait.h>
#include <luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h>
#include <luthier/ToolCodeGen/PrototypeCallGraph.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/Support/TargetSelect.h>
#include <spdlog/spdlog.h>
#include <atomic>
#include <map>
#include <mutex>
#include <set>

namespace bpftime::attach
{
namespace
{
using CoreSnapshot = luthier::rocprofiler::HsaApiTableSnapshot<CoreApiTable>;
using AmdSnapshot = luthier::rocprofiler::HsaApiTableSnapshot<AmdExtTable>;
using LoaderSnapshot =
	luthier::rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>;

void diagnose(llvm::Error error) noexcept
{
	if (!error)
		return;
	try {
		SPDLOG_WARN("ROCm instrumentation: {}",
			    llvm::toString(std::move(error)));
	} catch (...) {
		llvm::consumeError(std::move(error));
	}
}

// Compose the pipeline from the data/loader components, using bpftime's own
// recoverable API interception rather than Luthier's fatal-error traits.
struct Tool
	: luthier::Singleton<Tool>,
	  luthier::LoadedCodeObjectCache,
	  luthier::InstrumentedKernelLoaderAndLauncher,
	  luthier::InstrumentationPipelineTrait<Tool, llvm::MachineFunction> {
	Tool(CreationKey, const CoreSnapshot &core, const AmdSnapshot &amd,
	     const LoaderSnapshot &loader)
		: luthier::Singleton<Tool>(),
		  LoadedCodeObjectCache(core, loader),
		  InstrumentedKernelLoaderAndLauncher(core, amd, loader),
		  InstrumentationPipelineTrait(core, amd, loader)
	{
	}
	rocm_probe_snapshot compiling;
	std::string error;
	std::map<std::pair<uint64_t, uint64_t>, uint64_t> ready, failed;
	uint64_t last_generation = 0;
	std::atomic<unsigned> pending{ 0 };

	std::unique_ptr<llvm::Module>
	createInstrumentationModule(llvm::LLVMContext &ctx)
	{
		ctx.setDiagnosticHandlerCallBack(
			[](const llvm::DiagnosticInfo *info, void *data) {
				if (info->getSeverity() != llvm::DS_Error)
					return;
				auto &message =
					*static_cast<std::string *>(data);
				llvm::raw_string_ostream stream(message);
				llvm::DiagnosticPrinterRawOStream printer(
					stream);
				info->print(printer);
			},
			&error);
		auto module = create_rocm_module(compiling, ctx);
		if (!module) {
			error = llvm::toString(module.takeError());
			return std::make_unique<llvm::Module>("invalid.probe",
							      ctx);
		}
		return std::move(*module);
	}
	llvm::PreservedAnalyses run(luthier::Prototype &p,
				    luthier::PrototypeAnalysisManager &am)
	{
		if (!error.empty())
			return llvm::PreservedAnalyses::all();
		auto &module = p.getInstrumentationModule();
		auto &fam =
			am.getResult<
				  luthier::IModuleFunctionAnalysisManagerPrototypeProxy>(
				  p)
				.getManager();
		unsigned sites = 0;
		p.forEachTargetMF(am, [&](llvm::MachineFunction &mf) {
			if (mf.empty())
				return;
			llvm::SmallVector<llvm::MachineInstr *> exits;
			for (auto &bb : mf)
				for (auto &mi : bb)
					if (mf.getSubtarget()
						    .getInstrInfo()
						    ->getName(mi.getOpcode()) ==
					    "S_ENDPGM")
						exits.push_back(&mi);
			auto *entry = &mf.front().front();
			for (const auto &probe : compiling.probes) {
				if (mf.getName() != probe->kernel)
					continue;
				auto insert = [&](llvm::MachineInstr &mi) {
					llvm::IRBuilder<> b(
						module.getContext());
					auto payload = p.createInjectedPayload(
						*module.getFunction(
							probe->entry),
						mi, fam,
						{ llvm::ConstantPointerNull::get(
							b.getPtrTy()) },
						luthier::PayloadLaneMode::
							SingleLane);
					if (!payload)
						error += llvm::toString(
							payload.takeError());
					else
						++sites;
				};
				if (probe->at_return) {
					if (exits.empty())
						error = "Kernel has no supported return site";
					for (auto *mi : exits)
						insert(*mi);
				} else
					insert(*entry);
			}
		});
		if (!sites && error.empty())
			error = "No matching kernel instrumentation sites";
		auto pa = llvm::PreservedAnalyses::none();
		pa.preserve<luthier::TargetModuleAnalysisManagerPrototypeProxy>();
		pa.preserve<
			luthier::TargetFunctionAnalysisManagerPrototypeProxy>();
		pa.preserve<
			luthier::TargetMachineFunctionAnalysisManagerPrototypeProxy>();
		pa.preserve<luthier::IModuleAnalysisManagerPrototypeProxy>();
		pa.preserve<
			luthier::IModuleFunctionAnalysisManagerPrototypeProxy>();
		pa.preserve<
			luthier::IModuleMachineFunctionAnalysisManagerPrototypeProxy>();
		pa.abandon<luthier::InjectedPayloadAndInstPointAnalysis>();
		pa.abandon<luthier::PrototypeCallGraphAnalysis>();
		return pa;
	}
	void invalidate(hsa_executable_t exec)
	{
		// HSA requires the application to finish using this executable
		// before destroying it. Other executables may still be in
		// flight.
		diagnose(invalidateOriginalExec(exec));
		std::erase_if(ready, [&](const auto &item) {
			return item.second == exec.handle;
		});
		std::erase_if(failed, [&](const auto &item) {
			return item.second == exec.handle;
		});
		std::lock_guard lock(CacheMutex);
		LCOCache.clear();
		LoadedBaseToLCOMap.clear();
	}
};

struct state {
	std::recursive_mutex mutex;
	std::unique_ptr<CoreSnapshot> core;
	std::unique_ptr<AmdSnapshot> amd;
	std::unique_ptr<LoaderSnapshot> loader;
	const luthier::DynamicLibrary *library = nullptr;
	Tool *tool = nullptr;
	bool disabled = false;
};
state &get_state()
{
	static auto *s = new state;
	return *s;
}

bool initialize(state &s)
{
	if (s.disabled)
		return false;
	if (s.tool)
		return true;
	if (!s.core || !s.amd)
		return false;
	auto core = s.core->getTable();
	using Table = luthier::hsa::ExtensionApiTableInfo<
		HSA_EXTENSION_AMD_LOADER>::TableType;
	Table table{};
	if (core.callFunction<hsa_system_get_major_extension_table>(
		    HSA_EXTENSION_AMD_LOADER,
		    luthier::hsa::ExtensionApiTableInfo<
			    HSA_EXTENSION_AMD_LOADER>::MajorVer,
		    sizeof(table), &table) != HSA_STATUS_SUCCESS)
		return false;
	if (!table.hsa_ven_amd_loader_query_executable)
		return false;
	s.loader = std::make_unique<LoaderSnapshot>(table, *s.library);
	LLVMInitializeAMDGPUTargetInfo();
	LLVMInitializeAMDGPUTarget();
	LLVMInitializeAMDGPUTargetMC();
	LLVMInitializeAMDGPUAsmPrinter();
	LLVMInitializeAMDGPUAsmParser();
	LLVMInitializeAMDGPUDisassembler();
	LLVMInitializeAMDGPUTargetMCA();
	Tool::createInstance(*s.core, *s.amd, *s.loader);
	Tool::withInstance([&](Tool &t) { s.tool = &t; });
	return s.tool != nullptr;
}

struct dispatch_lifetime {
	luthier::ExtendedKernargBuffer arguments;
	hsa_signal_t proxy, original;
	Tool *tool;
	const CoreSnapshot *core;
};
bool complete(hsa_signal_value_t, void *data) noexcept
{
	auto *lifetime = static_cast<dispatch_lifetime *>(data);
	const auto core = lifetime->core->getTable();
	const auto original = lifetime->original;
	// Free resources before signaling application completion. No compiler
	// locks are acquired here; printf's hostcall listener remains
	// independent.
	llvm::consumeError(lifetime->arguments.release());
	core.callFunction<hsa_signal_destroy>(lifetime->proxy);
	lifetime->tool->pending.fetch_sub(1);
	delete lifetime;
	if (original.handle)
		core.callFunction<hsa_signal_subtract_screlease>(original, 1);
	return false;
}

bool prepare(state &s, const hsa_queue_t &queue,
	     hsa_kernel_dispatch_packet_t &packet)
{
	auto snapshot = snapshot_rocm_probes();
	if (snapshot.probes.empty() || !initialize(s))
		return false;
	auto &tool = *s.tool;
	auto core = s.core->getTable();
	auto amd = s.amd->getTable();
	const auto *kd =
		reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
			packet.kernel_object);
	hsa_executable_t exec{};
	hsa_amd_pointer_info_t info{};
	info.size = sizeof(info);
	if (s.loader->getTable().hsa_ven_amd_loader_query_executable(
		    kd, &exec) != HSA_STATUS_SUCCESS ||
	    amd.callFunction<hsa_amd_pointer_info>(
		    const_cast<void *>(static_cast<const void *>(kd)), &info,
		    nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS)
		return false;
	std::erase_if(snapshot.probes, [&](const auto &probe) {
		for (const auto &name :
		     { probe->kernel + ".kd", probe->kernel }) {
			hsa_executable_symbol_t symbol{};
			uint64_t object = 0;
			if (core.callFunction<hsa_executable_get_symbol_by_name>(
				    exec, name.c_str(), &info.agentOwner,
				    &symbol) == HSA_STATUS_SUCCESS &&
			    core.callFunction<hsa_executable_symbol_get_info>(
				    symbol,
				    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
				    &object) == HSA_STATUS_SUCCESS &&
			    object == packet.kernel_object)
				return false;
		}
		return true;
	});
	if (snapshot.probes.empty())
		return false;
	// Reclaim superseded probe generations only between dispatches. A live
	// dispatch owns its extended arguments and still references the old
	// code.
	if (tool.last_generation != snapshot.generation &&
	    !tool.pending.load()) {
		diagnose(tool.unloadAll());
		tool.ready.clear();
		tool.failed.clear();
		tool.last_generation = snapshot.generation;
	}
	const auto key =
		std::make_pair(packet.kernel_object, snapshot.generation);
	if (tool.failed.contains(key))
		return false;
	if (!tool.ready.contains(key)) {
		auto tm = tool.buildTargetMachineForKD(kd);
		if (!tm) {
			diagnose(tm.takeError());
			tool.failed.emplace(key, exec.handle);
			return false;
		}
		if ((*tm)->getTargetCPU() != BPFTIME_ROCM_HELPER_ARCH) {
			SPDLOG_WARN(
				"ROCm helper was built for {}, dispatch uses {}",
				BPFTIME_ROCM_HELPER_ARCH,
				(*tm)->getTargetCPU().str());
			tool.failed.emplace(key, exec.handle);
			return false;
		}
		tool.compiling = snapshot;
		tool.error.clear();
		auto object = tool.runInstrumentationPipelineForDispatch(*kd);
		if (!object || !tool.error.empty()) {
			if (!object)
				diagnose(object.takeError());
			if (!tool.error.empty())
				SPDLOG_WARN("ROCm instrumentation: {}",
					    tool.error);
			tool.failed.emplace(key, exec.handle);
			return false;
		}
		auto loaded = tool.loadInstrumented(std::move(*object), kd,
						    snapshot.generation);
		if (!loaded) {
			diagnose(loaded.takeError());
			tool.failed.emplace(key, exec.handle);
			return false;
		}
		tool.ready.emplace(key, exec.handle);
		SPDLOG_DEBUG(
			"ROCm instrumented kernel cached: object={:#x}, generation={}",
			key.first, key.second);
	}
	auto args = tool.overrideWithInstrumented(packet, queue,
						  snapshot.generation);
	if (!args) {
		diagnose(args.takeError());
		return false;
	}
	auto lifetime = std::make_unique<dispatch_lifetime>();
	lifetime->arguments = std::move(*args);
	lifetime->original = packet.completion_signal;
	lifetime->tool = &tool;
	lifetime->core = s.core.get();
	if (amd.callFunction<hsa_amd_signal_create>(
		    1, 0, nullptr, 0, &lifetime->proxy) != HSA_STATUS_SUCCESS)
		return false;
	tool.pending.fetch_add(1);
	if (amd.callFunction<hsa_amd_signal_async_handler>(
		    lifetime->proxy, HSA_SIGNAL_CONDITION_EQ, 0, complete,
		    lifetime.get()) != HSA_STATUS_SUCCESS) {
		tool.pending.fetch_sub(1);
		core.callFunction<hsa_signal_destroy>(lifetime->proxy);
		return false;
	}
	packet.completion_signal = lifetime->proxy;
	lifetime.release();
	return true;
}
} // namespace

void capture_rocm_tables(const HsaApiTable &root,
			 const luthier::DynamicLibrary &library) noexcept
{
	try {
		auto &s = get_state();
		std::lock_guard lock(s.mutex);
		if (s.core)
			return;
		// Optional instrumentation entries must all be available;
		// observing dispatches remains possible with smaller tables
		// (e.g. the mock SDK).
		CoreApiTable core{};
		AmdExtTable amd{};
		core.version = { HSA_CORE_API_TABLE_MAJOR_VERSION,
				 sizeof(CoreApiTable),
				 HSA_CORE_API_TABLE_STEP_VERSION };
		amd.version = { HSA_AMD_EXT_API_TABLE_MAJOR_VERSION,
				sizeof(AmdExtTable),
				HSA_AMD_EXT_API_TABLE_STEP_VERSION };
		copyElement(&core.version, &root.core_->version);
		copyElement(&amd.version, &root.amd_ext_->version);
		if (!core.hsa_system_get_major_extension_table_fn ||
		    !core.hsa_executable_get_symbol_by_name_fn ||
		    !core.hsa_executable_symbol_get_info_fn ||
		    !core.hsa_signal_destroy_fn ||
		    !core.hsa_signal_subtract_screlease_fn ||
		    !amd.hsa_amd_signal_async_handler_fn ||
		    !amd.hsa_amd_signal_create_fn ||
		    !amd.hsa_amd_pointer_info_fn)
			return;
		auto core_snapshot =
			std::make_unique<CoreSnapshot>(core, library);
		auto amd_snapshot = std::make_unique<AmdSnapshot>(amd, library);
		s.core = std::move(core_snapshot);
		s.amd = std::move(amd_snapshot);
		s.library = &library;
	} catch (...) {
	}
}
void instrument_rocm_packets(
	const hsa_queue_t &queue, const void *packets, uint64_t count,
	hsa_amd_queue_intercept_packet_writer writer) noexcept
{
	const auto *items =
		static_cast<const luthier::hsa::AqlPacket *>(packets);
	for (uint64_t i = 0; i < count; ++i) {
		auto packet = items[i];
		try {
			if (auto *dispatch = packet.asKernelDispatch()) {
				auto &s = get_state();
				std::lock_guard lock(s.mutex);
				if (!prepare(s, queue, *dispatch))
					packet = items[i];
			}
		} catch (const std::exception &e) {
			packet = items[i];
			try {
				SPDLOG_WARN(
					"ROCm instrumentation disabled for dispatch: {}",
					e.what());
			} catch (...) {
			}
		} catch (...) {
			packet = items[i];
		}
		writer(&packet, 1);
	}
}
void invalidate_rocm_executable(hsa_executable_t exec) noexcept
{
	try {
		auto &s = get_state();
		std::lock_guard lock(s.mutex);
		if (s.tool)
			s.tool->invalidate(exec);
	} catch (...) {
	}
}
void shutdown_rocm_instrumentation() noexcept
{
	try {
		auto &s = get_state();
		std::lock_guard lock(s.mutex);
		s.disabled = true;
		// Outstanding dispatches retain their code, hostcall service
		// and args.
		if (s.tool && !s.tool->pending.load()) {
			Tool::destroyInstance();
			s.tool = nullptr;
		}
	} catch (...) {
	}
}
void suspend_rocm_instrumentation() noexcept
{
	try {
		auto &s = get_state();
		std::lock_guard lock(s.mutex);
		// HSA owns the listener's signal. Stop and destroy the tool
		// before the runtime tears down its signals; a later HSA init
		// can rebuild it.
		if (s.tool && !s.tool->pending.load()) {
			Tool::destroyInstance();
			s.tool = nullptr;
		}
	} catch (...) {
	}
}
} // namespace bpftime::attach
