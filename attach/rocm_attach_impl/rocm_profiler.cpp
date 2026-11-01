// SPDX-License-Identifier: MIT
#include "rocm_profiler.hpp"
#include <luthier/HSA/AqlPacket.h>
#include <luthier/HSA/ApiTable.h>
#include <luthier/Rocprofiler/RocprofilerSdkLibrary.h>
#include <rocprofiler-sdk/version.h>
#include <spdlog/spdlog.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace bpftime::attach
{
namespace
{
struct queue_api {
	decltype(hsa_queue_create) *create = nullptr;
	decltype(hsa_queue_destroy) *destroy = nullptr;
	decltype(hsa_amd_queue_intercept_create) *intercept_create = nullptr;
	decltype(hsa_amd_queue_intercept_register) *intercept_register =
		nullptr;
#ifdef HSA_AMD_QUEUE_CREATE_DESC_VERSION
	decltype(hsa_amd_queue_create) *create_desc = nullptr;
	decltype(hsa_amd_queue_set_priority) *set_priority = nullptr;
	decltype(hsa_amd_queue_cu_set_mask) *set_mask = nullptr;
#endif
};

struct profiler_state {
	std::mutex mutex;
	std::unique_ptr<luthier::DynamicLibrary> sdk;
	// API wrappers and registered queue callbacks can outlive tool
	// finalization. Retain their storage until process exit, including
	// across HSA reinit.
	std::vector<std::unique_ptr<queue_api>> tables;
	std::atomic<const queue_api *> api{ nullptr };
	std::atomic<bool> enabled{ false };
	std::atomic<uint64_t> dispatches{ 0 };
	bool configured = false;
	bool started = false;
	bool finalized = false;
	bool (*initialize_agent)() noexcept = nullptr;
	void (*finalize_agent)() noexcept = nullptr;
};

profiler_state &state()
{
	static auto *value = new profiler_state;
	return *value;
}

void diagnose(const char *message) noexcept
{
	try {
		SPDLOG_WARN("ROCm profiler: {}", message);
	} catch (...) {
	}
}

void on_packets(const void *packets, uint64_t count, uint64_t, void *data,
		hsa_amd_queue_intercept_packet_writer writer)
{
	auto &s = *static_cast<profiler_state *>(data);
	try {
		if (s.enabled.load(std::memory_order_acquire)) {
			const auto *items =
				static_cast<const luthier::hsa::AqlPacket *>(
					packets);
			for (uint64_t i = 0; i < count; ++i) {
				if (auto *dispatch =
					    items[i].asKernelDispatch()) {
					s.dispatches.fetch_add(
						1, std::memory_order_relaxed);
					SPDLOG_DEBUG(
						"ROCm dispatch: kernel_object={:#x}, grid=({},{},{}), workgroup=({},{},{})",
						dispatch->kernel_object,
						dispatch->grid_size_x,
						dispatch->grid_size_y,
						dispatch->grid_size_z,
						dispatch->workgroup_size_x,
						dispatch->workgroup_size_y,
						dispatch->workgroup_size_z);
				}
			}
		}
	} catch (...) {
		// Instrumentation failure must not drop or replay application
		// packets.
		diagnose(
			"dispatch observation failed; forwarding original packets");
	}
	writer(packets, count);
}

// Keep the original queue alive until its replacement is fully configured.
// Luthier's current PacketMonitorTrait destroys it first and uses fatal-error
// handling on failure; that is unsuitable for an injected bpftime agent.
hsa_status_t create_queue(hsa_agent_t agent, uint32_t size,
			  hsa_queue_type32_t type,
			  void (*callback)(hsa_status_t, hsa_queue_t *, void *),
			  void *data, uint32_t private_size,
			  uint32_t group_size, hsa_queue_t **queue)
{
	auto &s = state();
	const auto *api = s.api.load(std::memory_order_acquire);
	if (!api || !api->create)
		return HSA_STATUS_ERROR_NOT_INITIALIZED;
	auto status = api->create(agent, size, type, callback, data,
				  private_size, group_size, queue);
	if (status != HSA_STATUS_SUCCESS || !queue || !*queue ||
	    !s.enabled.load(std::memory_order_acquire))
		return status;
	auto registered = api->intercept_register(*queue, on_packets, &s);
	if (registered == HSA_STATUS_SUCCESS)
		return status;
	if (registered != HSA_STATUS_ERROR_INVALID_QUEUE) {
		diagnose("cannot monitor queue; preserving original queue");
		return status;
	}
	hsa_queue_t *replacement = nullptr;
	auto result =
		api->intercept_create(agent, size, type, callback, data,
				      private_size, group_size, &replacement);
	if (result == HSA_STATUS_SUCCESS) {
		result = api->intercept_register(replacement, on_packets, &s);
		if (result == HSA_STATUS_SUCCESS)
			result = api->destroy(*queue);
		if (result == HSA_STATUS_SUCCESS)
			*queue = replacement;
		else
			(void)api->destroy(replacement);
	}
	if (result != HSA_STATUS_SUCCESS)
		diagnose(
			"cannot create intercept queue; preserving original queue");
	return status;
}

#ifdef HSA_AMD_QUEUE_CREATE_DESC_VERSION
hsa_status_t create_queues(hsa_agent_t agent,
			   hsa_amd_queue_create_desc_t *descs, uint32_t count)
{
	auto &s = state();
	const auto *api = s.api.load(std::memory_order_acquire);
	if (!api || !api->create_desc)
		return HSA_STATUS_ERROR_NOT_INITIALIZED;
	auto status = api->create_desc(agent, descs, count);
	if (status != HSA_STATUS_SUCCESS || !descs ||
	    !s.enabled.load(std::memory_order_acquire))
		return status;
	for (uint32_t i = 0; i < count; ++i) {
		auto &d = descs[i];
		if (!d.queue ||
		    d.version != HSA_AMD_QUEUE_CREATE_DESC_VERSION ||
		    d.engine_type != HSA_AMD_QUEUE_ENGINE_COMPUTE ||
		    d.flags != 0)
			continue;
		auto result = api->intercept_register(d.queue, on_packets, &s);
		if (result == HSA_STATUS_SUCCESS)
			continue;
		if (result != HSA_STATUS_ERROR_INVALID_QUEUE) {
			diagnose(
				"cannot monitor descriptor queue; preserving original queue");
			continue;
		}
		hsa_queue_t *replacement = nullptr;
		result = api->intercept_create(
			agent,
			d.queue_size_bytes / sizeof(luthier::hsa::AqlPacket),
			d.engine.compute.type, d.callback, d.callback_data,
			d.engine.compute.private_segment_size, UINT32_MAX,
			&replacement);
		if (result == HSA_STATUS_SUCCESS) {
			result = api->intercept_register(replacement,
							 on_packets, &s);
			if (result == HSA_STATUS_SUCCESS)
				result = api->set_priority(replacement,
							   d.priority);
			if (result == HSA_STATUS_SUCCESS &&
			    d.engine.compute.cu_mask_count)
				result = api->set_mask(
					replacement,
					d.engine.compute.cu_mask_count,
					d.engine.compute.cu_mask);
			if (result == HSA_STATUS_SUCCESS)
				result = api->destroy(d.queue);
			if (result == HSA_STATUS_SUCCESS)
				d.queue = replacement;
			else
				(void)api->destroy(replacement);
		}
		if (result != HSA_STATUS_SUCCESS)
			diagnose(
				"cannot replace descriptor queue; preserving original queue");
	}
	return status;
}
#endif

// Use Luthier's table layout checks and packet types, with recoverable errors
// at the profiler boundary instead of the traits' fatal-error handlers.
void register_tables(rocprofiler_intercept_table_t type, uint64_t, uint64_t,
		     void **tables, uint64_t count, void *data) noexcept
{
	try {
		auto &s = *static_cast<profiler_state *>(data);
		std::lock_guard guard(s.mutex);
		if (s.finalized)
			return;
		if (type != ROCPROFILER_HSA_TABLE || count != 1 || !tables ||
		    !tables[0]) {
			diagnose(
				"invalid HSA registration; instrumentation disabled for this runtime");
			return;
		}
		auto &root = *static_cast<HsaApiTable *>(tables[0]);
		using luthier::hsa::apiTableHasEntry;
		if (root.version.major_id != HSA_API_TABLE_MAJOR_VERSION ||
		    !apiTableHasEntry<CoreApiTable>(root) || !root.core_ ||
		    !apiTableHasEntry<AmdExtTable>(root) || !root.amd_ext_) {
			diagnose(
				"unsupported HSA table; instrumentation disabled for this runtime");
			return;
		}
		auto &core = *root.core_;
		auto &amd = *root.amd_ext_;
		if (core.version.major_id != HSA_CORE_API_TABLE_MAJOR_VERSION ||
		    amd.version.major_id != HSA_AMD_EXT_API_TABLE_MAJOR_VERSION ||
		    !apiTableHasEntry<&CoreApiTable::hsa_queue_create_fn>(
			    core) ||
		    !apiTableHasEntry<&CoreApiTable::hsa_queue_destroy_fn>(
			    core) ||
		    !apiTableHasEntry<
			    &AmdExtTable::hsa_amd_queue_intercept_create_fn>(
			    amd) ||
		    !apiTableHasEntry<
			    &AmdExtTable::hsa_amd_queue_intercept_register_fn>(
			    amd) ||
		    !core.hsa_queue_create_fn || !core.hsa_queue_destroy_fn ||
		    !amd.hsa_amd_queue_intercept_create_fn ||
		    !amd.hsa_amd_queue_intercept_register_fn) {
			diagnose(
				"HSA queue interception unavailable; preserving runtime tables");
			return;
		}
		if (core.hsa_queue_create_fn == create_queue)
			return;
		auto api = std::make_unique<queue_api>();
		api->create = core.hsa_queue_create_fn;
		api->destroy = core.hsa_queue_destroy_fn;
		api->intercept_create = amd.hsa_amd_queue_intercept_create_fn;
		api->intercept_register =
			amd.hsa_amd_queue_intercept_register_fn;
#ifdef HSA_AMD_QUEUE_CREATE_DESC_VERSION
		if (apiTableHasEntry(amd, &AmdExtTable::hsa_amd_queue_create_fn) &&
		    apiTableHasEntry<&AmdExtTable::hsa_amd_queue_set_priority_fn>(
			    amd) &&
		    apiTableHasEntry<&AmdExtTable::hsa_amd_queue_cu_set_mask_fn>(
			    amd) &&
		    amd.hsa_amd_queue_create_fn &&
		    amd.hsa_amd_queue_set_priority_fn &&
		    amd.hsa_amd_queue_cu_set_mask_fn) {
			api->create_desc = amd.hsa_amd_queue_create_fn;
			api->set_priority = amd.hsa_amd_queue_set_priority_fn;
			api->set_mask = amd.hsa_amd_queue_cu_set_mask_fn;
		}
#endif
		const auto *saved = api.get();
		s.tables.push_back(std::move(api));
		s.api.store(saved, std::memory_order_release);
		core.hsa_queue_create_fn = create_queue;
#ifdef HSA_AMD_QUEUE_CREATE_DESC_VERSION
		if (saved->create_desc)
			amd.hsa_amd_queue_create_fn = create_queues;
#endif
	} catch (...) {
		diagnose(
			"cannot install queue interceptors; preserving runtime tables");
	}
}

int initialize(rocprofiler_client_finalize_t, void *data) noexcept
{
	auto &s = *static_cast<profiler_state *>(data);
	try {
		std::lock_guard guard(s.mutex);
		if (s.finalized)
			return -1;
		if (s.started)
			return 0;
		s.started = true;
		s.enabled.store(true, std::memory_order_release);
	} catch (...) {
		return -1;
	}
	const bool attached = s.initialize_agent && s.initialize_agent();
	try {
		SPDLOG_INFO("ROCm profiler initialized (agent session: {})",
			    attached ? "ready" : "unavailable");
	} catch (...) {
	}
	return 0;
}

void finalize(void *data) noexcept
{
	auto &s = *static_cast<profiler_state *>(data);
	try {
		std::lock_guard guard(s.mutex);
		if (s.finalized)
			return;
		s.finalized = true;
		s.enabled.store(false, std::memory_order_release);
	} catch (...) {
		return;
	}
	if (s.finalize_agent)
		s.finalize_agent();
	try {
		SPDLOG_INFO("ROCm profiler finalized: {} dispatches",
			    s.dispatches.load());
	} catch (...) {
	}
}
} // namespace

rocprofiler_tool_configure_result_t *
configure_rocm_profiler(uint32_t version, rocprofiler_client_id_t *client,
			bool (*initialize_agent)() noexcept,
			void (*finalize_agent)() noexcept) noexcept
{
	try {
		if (!client ||
		    version / 10000 != ROCPROFILER_SDK_VERSION_MAJOR) {
			diagnose("incompatible ROCprofiler SDK version");
			return nullptr;
		}
		auto &s = state();
		std::lock_guard guard(s.mutex);
		if (s.finalized)
			return nullptr;
		static rocprofiler_tool_configure_result_t config{
			sizeof(config), initialize, finalize, &s
		};
		client->name = "bpftime ROCm agent";
		if (s.configured)
			return &config;
		auto library =
			luthier::rocprofiler::openRocprofilerSdkLibrary();
		if (!library) {
			llvm::consumeError(library.takeError());
			diagnose("cannot open ROCprofiler SDK");
			return nullptr;
		}
		s.sdk = std::make_unique<luthier::DynamicLibrary>(
			std::move(*library));
		// A function-address template argument emits a direct SDK symbol
		// reference in Clang's debug info. Resolve by name so Debug builds
		// also depend on the SDK only at runtime.
		auto registration = reinterpret_cast<
			decltype(&rocprofiler_at_intercept_table_registration)>(
			s.sdk->getSymbol<void>(
				"rocprofiler_at_intercept_table_registration"));
		if (!registration ||
		    registration(register_tables, ROCPROFILER_HSA_TABLE, &s) !=
			    ROCPROFILER_STATUS_SUCCESS) {
			diagnose("cannot register HSA table callback");
			return nullptr;
		}
		s.initialize_agent = initialize_agent;
		s.finalize_agent = finalize_agent;
		s.configured = true;
		return &config;
	} catch (...) {
		diagnose("configuration failed; instrumentation disabled");
		return nullptr;
	}
}
} // namespace bpftime::attach
