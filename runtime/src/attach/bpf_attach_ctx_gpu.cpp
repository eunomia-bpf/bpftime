#include "bpftime_shm_internal.hpp"
#include <bpf_attach_ctx.hpp>
#include <spdlog/spdlog.h>
#ifdef BPFTIME_ENABLE_ROCM_ATTACH
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
extern "C" uint64_t bpftime_trace_printk(uint64_t fmt, uint64_t fmt_size, ...);
#endif

namespace bpftime
{
std::vector<attach::MapBasicInfo>
bpf_attach_ctx::create_map_basic_info(int filled_size)
{
	std::vector<attach::MapBasicInfo> local_basic_info(filled_size);
	for (auto &entry : local_basic_info) {
		entry.enabled = false;
		entry.key_size = 0;
		entry.value_size = 0;
		entry.max_entries = 0;
		entry.map_type = 0;
		entry.extra_buffer = nullptr;
		entry.max_thread_count = 0;
	}
	const auto &handler_manager =
		*shm_holder.global_shared_memory.get_manager();
	for (size_t i = 0; i < handler_manager.size(); i++) {
		const auto &current_handler = handler_manager.get_handler(i);
		if (std::holds_alternative<bpf_map_handler>(current_handler)) {
			if (i >= local_basic_info.size()) {
				SPDLOG_ERROR(
					"Too large map fd: {}, max to be {}", i,
					local_basic_info.size());
				return {};
			}
			auto &local = local_basic_info[i];
			const auto &map =
				std::get<bpf_map_handler>(current_handler);
			void *gpu_buffer = nullptr;
#ifdef BPFTIME_ENABLE_CUDA_ATTACH
			gpu_buffer = map.get_gpu_map_extra_buffer();
#endif
			SPDLOG_INFO(
				"Copying map fd {} to device, key size={}, value size={}, max ent={}, map_type = {}, gpu_buffer = {:x}",
				i, map.get_key_size(), map.get_value_size(),
				map.get_max_entries(), (int)map.type,
				(uintptr_t)gpu_buffer);

			local.enabled = true;
			local.key_size = map.get_key_size();
			local.value_size = map.get_value_size();
			local.max_entries = map.get_max_entries();
			local.map_type = (int)map.type;
			local.extra_buffer = gpu_buffer;
#ifdef BPFTIME_ENABLE_CUDA_ATTACH
			local.max_thread_count =
				map.get_gpu_map_max_thread_count();
#endif
		}
	}

	return local_basic_info;
}

#ifdef BPFTIME_ENABLE_ROCM_ATTACH
void bpf_attach_ctx::start_gpu_watcher_thread()
{
	if (!gpu_ctx || gpu_watcher_thread.joinable())
		return;
	auto flag = this->gpu_ctx->cuda_watcher_should_stop;
	gpu_watcher_thread = std::thread([flag, this]() {
		SPDLOG_INFO("ROCm watcher thread started");
		auto &ctx = gpu_ctx;

		while (!flag->load()) {
			if (ctx->shared_mem->flag1 == 1) {
				ctx->shared_mem->flag1 = 0;
				auto req_id = ctx->shared_mem->request_id;

				auto map_ptr = ctx->shared_mem->map_id;
				auto map_fd = map_ptr;
				SPDLOG_DEBUG(
					"ROCm Received call request id {}, map_ptr = {}, map_fd = {}",
					req_id, map_ptr, map_fd);
				auto start_time =
					std::chrono::high_resolution_clock::now();
				if (req_id ==
				    (int)gpu::HelperOperation::MAP_LOOKUP) {
					const auto &req =
						ctx->shared_mem->req.map_lookup;
					auto &resp =
						ctx->shared_mem->resp.map_lookup;
					auto ptr = bpftime_map_lookup_elem(
						map_fd, req.key);
					resp.value = ptr;
					SPDLOG_DEBUG(
						"ROCm: Executing map lookup for {}, key= {:x} result = {:x}",
						map_fd,
						*(uintptr_t *)(uintptr_t)&req
							 .key,
						(uintptr_t)resp.value);

				} else if (req_id ==
					   (int)gpu::HelperOperation::MAP_UPDATE) {
					const auto &req =
						ctx->shared_mem->req.map_update;
					auto &resp =
						ctx->shared_mem->resp.map_update;
					resp.result = bpftime_map_update_elem(
						map_fd, req.key, req.value,
						req.flags);
					SPDLOG_DEBUG(
						"ROCm: Executing map update for {}, result = {}",
						map_fd, resp.result);
				} else if (req_id ==
					   (int)gpu::HelperOperation::MAP_DELETE) {
					const auto &req =
						ctx->shared_mem->req.map_delete;
					auto &resp =
						ctx->shared_mem->resp.map_delete;

					resp.result = bpftime_map_delete_elem(
						map_fd, req.key);
					SPDLOG_DEBUG(
						"ROCm: Executing map delete for {}, result = {}",
						map_fd, resp.result);
				} else if (req_id == (int)gpu::HelperOperation::
							     TRACE_PRINTK) {
					const auto &req = ctx->shared_mem->req
								  .trace_printk;
					auto &resp = ctx->shared_mem->resp
							     .trace_printk;

					resp.result = bpftime_trace_printk(
						(uintptr_t)req.fmt,
						req.fmt_size, req.arg1,
						req.arg2, req.arg3);
					SPDLOG_DEBUG(
						"ROCm: Executing bpf_printk: {}, arg1={}, arg2={}, arg3={}",
						req.fmt, req.arg1, req.arg2,
						req.arg3);
				} else if (req_id ==
					   (int)gpu::HelperOperation::
						   GET_CURRENT_PID_TGID) {
					auto &resp = ctx->shared_mem->resp
							     .get_tid_pgid;
					static int tgid = getpid();
					static thread_local int tid = -1;
					if (tid == -1) {
						tid = gettid();
					}
					SPDLOG_DEBUG(
						"Called get_current_pid_tgid: pid={}, tgid={}",
						tid, tgid);
					resp.result =
						(((uint64_t)tgid) << 32) | tid;
				} else if (req_id ==
					   (int)gpu::HelperOperation::PUTS) {
					const auto &req =
						ctx->shared_mem->req.puts;
					auto &resp = ctx->shared_mem->resp.puts;
					SPDLOG_INFO("eBPF: {}", req.data);
					resp.result = 0;
				}

				else {
					SPDLOG_WARN("Unknown request id {}",
						    req_id);
				}

				std::atomic_thread_fence(
					std::memory_order_seq_cst);
				ctx->shared_mem->flag2 = 1;
				std::atomic_thread_fence(
					std::memory_order_seq_cst);
			}
			std::this_thread::sleep_for(
				std::chrono::milliseconds(1));
		}
		SPDLOG_INFO("Exiting ROCm watcher thread");
	});
}

void bpf_attach_ctx::stop_gpu_watcher_thread()
{
	if (gpu_ctx)
		gpu_ctx->cuda_watcher_should_stop->store(
			true, std::memory_order_release);
	if (gpu_watcher_thread.joinable())
		gpu_watcher_thread.join();
}

namespace gpu
{
std::optional<std::unique_ptr<GPUContext>> create_gpu_context()
{
	auto mem = std::make_unique<CommSharedMem>();
	if (auto err = hipHostRegister(mem.get(), sizeof(CommSharedMem),
				       hipHostRegisterMapped);
	    err != hipSuccess) {
		SPDLOG_WARN("Unable to register HIP shared memory: {}",
			    (int)err);
		return {};
	}
	void *device_ptr = nullptr;
	if (auto err = hipHostGetDevicePointer(&device_ptr, mem.get(), 0);
	    err != hipSuccess) {
		hipHostUnregister(mem.get());
		SPDLOG_WARN("Unable to map HIP shared memory: {}", (int)err);
		return {};
	}
	try {
		auto ctx = std::make_unique<GPUContext>(std::move(mem));
		ctx->rocm_shared_mem_device_pointer =
			reinterpret_cast<uintptr_t>(device_ptr);
		return std::make_optional(std::move(ctx));
	} catch (...) {
		hipHostUnregister(mem.get());
		throw;
	}
}
GPUContext::GPUContext(std::unique_ptr<CommSharedMem> &&mem)
	: shared_mem(std::move(mem))
{
}
GPUContext::~GPUContext()
{
	if (auto err = hipHostUnregister(shared_mem.get()); err != hipSuccess)
		SPDLOG_WARN("Unable to unregister HIP shared memory: {}",
			    (int)err);
}
} // namespace gpu

#endif
} // namespace bpftime
