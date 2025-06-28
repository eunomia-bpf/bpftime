#include "bpftime_prog.hpp"
#include "bpftime_shm_internal.hpp"

#include <array>
#include <bpf_attach_ctx.hpp>
#include "demo_ptx_prog.hpp"
#include <memory>
#include <optional>
#include <spdlog/spdlog.h>
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#endif
#if defined(BPFTIME_ENABLE_ROCM_ATTACH)
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif
extern "C" {
extern uint64_t bpftime_trace_printk(uint64_t fmt, uint64_t fmt_size, ...);
}

#define CUDART_SAFE_CALL(x, error)                                             \
	do {                                                                   \
		if (auto result = x; result != cudaSuccess) {                  \
			SPDLOG_ERROR("error: {} failed with error code {}",    \
				     #x, (int)result);                         \
			throw std::runtime_error(error);                       \
		}                                                              \
	} while (0)
#define NV_SAFE_CALL(x, error_message)                                         \
	do {                                                                   \
		CUresult result = x;                                           \
		if (result != CUDA_SUCCESS) {                                  \
			SPDLOG_ERROR("error: {} failed with error code {}",    \
				     #x, (int)result);                         \
			throw std::runtime_error(error_message);               \
		}                                                              \
	} while (0)

#define NV_SAFE_CALL_NO_THROW(x, error_message)                                \
	do {                                                                   \
		CUresult result = x;                                           \
		if (result != CUDA_SUCCESS) {                                  \
			SPDLOG_ERROR("error: {} failed with error code {}",    \
				     #x, (int)result);                         \
		}                                                              \
	} while (0)

#define NV_SAFE_CALL_2(x, error_message)                                       \
	do {                                                                   \
		CUresult result = x;                                           \
		if (result != CUDA_SUCCESS) {                                  \
			SPDLOG_ERROR(                                          \
				"error: {} failed with error code {}: {}", #x, \
				(int)result, error_message);                   \
			return -1;                                             \
		}                                                              \
	} while (0)

#define NV_SAFE_CALL_3(x, error_message)                                       \
	do {                                                                   \
		CUresult result = x;                                           \
		if (result != CUDA_SUCCESS) {                                  \
			SPDLOG_ERROR(                                          \
				"error: {} failed with error code {}: {}", #x, \
				(int)result, error_message);                   \
			return {};                                             \
		}                                                              \
	} while (0)

namespace bpftime
{

void bpf_attach_ctx::start_gpu_watcher_thread()
{
	auto flag = this->gpu_ctx->cuda_watcher_should_stop;
	std::thread handle([=, this]() {
		SPDLOG_INFO("CUDA watcher thread started");
		auto &ctx = gpu_ctx;

		while (!flag->load()) {
			if (ctx->shared_mem->flag1 == 1) {
				ctx->shared_mem->flag1 = 0;
				auto req_id = ctx->shared_mem->request_id;

				auto map_ptr = ctx->shared_mem->map_id;
				auto map_fd = map_ptr;
				SPDLOG_DEBUG(
					"CUDA Received call request id {}, map_ptr = {}, map_fd = {}",
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
						"CUDA: Executing map lookup for {}, key= {:x} result = {:x}",
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
						"CUDA: Executing map update for {}, result = {}",
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
						"CUDA: Executing map delete for {}, result = {}",
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
						"CUDA: Executing bpf_printk: {}, arg1={}, arg2={}, arg3={}",
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

				ctx->shared_mem->flag2 = 1;
				std::atomic_thread_fence(
					std::memory_order_seq_cst);
			}
			std::this_thread::sleep_for(
				std::chrono::milliseconds(1));
		}
		SPDLOG_INFO("Exiting CUDA watcher thread");
	});
	handle.detach();
}
std::vector<attach::MapBasicInfo>
bpf_attach_ctx::create_map_basic_info(int filled_size)
{
	std::vector<attach::MapBasicInfo> local_basic_info(filled_size);
	for (auto &entry : local_basic_info) {
		entry.enabled = false;
		entry.key_size = 0;
		entry.value_size = 0;
		entry.max_entries = 0;
	}
	const auto &handler_manager =
		*shm_holder.global_shared_memory.get_manager();
	for (size_t i = 0; i < handler_manager.size(); i++) {
		const auto &current_handler = handler_manager.get_handler(i);
		if (std::holds_alternative<bpf_map_handler>(current_handler)) {
			auto &local = local_basic_info[i];
			SPDLOG_INFO(
				"Copying map fd {} to device, key size={}, value size={}, max ent={}",
				i, local.key_size, local.value_size,
				local.max_entries);
			const auto &map =
				std::get<bpf_map_handler>(current_handler);
			if (i >= local_basic_info.size()) {
				SPDLOG_ERROR(
					"Too large map fd: {}, max to be {}", i,
					local_basic_info.size());
				return {};
			}

			local.enabled = true;
			local.key_size = map.get_key_size();
			local.value_size = map.get_value_size();
			local.max_entries = map.get_max_entries();
		}
	}

	return local_basic_info;
}

namespace gpu
{
std::optional<std::unique_ptr<gpu::GPUContext>> create_gpu_context()
{
	SPDLOG_INFO("Initializing GPU shared memory");
	auto gpu_shared_mem = std::make_unique<gpu::CommSharedMem>();
	memset(gpu_shared_mem.get(), 0, sizeof(*gpu_shared_mem));
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
	CUDART_SAFE_CALL(cudaHostRegister(gpu_shared_mem.get(),
					  sizeof(gpu::CommSharedMem),
					  cudaHostRegisterDefault),
			 "Unable to register shared memory");
#endif
#if defined(BPFTIME_ENABLE_ROCM_ATTACH)
	if (auto err = hipHostRegister(gpu_shared_mem.get(),
				       sizeof(gpu::CommSharedMem),
				       hipHostRegisterDefault);
	    err != hipSuccess) {
		SPDLOG_ERROR("Unable to register HIP shared memory: {}",
			     (int)err);
		throw std::runtime_error(
			"Unable to register HIP shared memory");
	}
#endif

	auto cuda_ctx = std::make_optional(
		std::make_unique<gpu::GPUContext>(std::move(gpu_shared_mem)));

	SPDLOG_INFO("CUDA context created");
	return cuda_ctx;
}
GPUContext::~GPUContext()
{
	SPDLOG_INFO("Destructing CUDAContext");

#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
	if (auto result = cudaHostUnregister(cuda_shared_mem.get());
	    result != cudaSuccess) {
		SPDLOG_ERROR("Unable to unregister host memory: {}",
			     (int)result);
	}
#endif
}
GPUContext::GPUContext(std::unique_ptr<gpu::CommSharedMem> &&mem)
	: shared_mem(std::move(mem))

{
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
	cuda_shared_mem_device_pointer = (uintptr_t)shared_mem.get();
#endif
#if defined(BPFTIME_ENABLE_ROCM_ATTACH)
	rocm_shared_mem_device_pointer = (uintptr_t)shared_mem.get();
#endif
}

} // namespace gpu
} // namespace bpftime
