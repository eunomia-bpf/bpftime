#include "bpftime_prog.hpp"
#include "bpftime_shm_internal.hpp"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "nv_attach_impl.hpp"
#include <array>
#include <bpf_attach_ctx.hpp>
#include "demo_ptx_prog.hpp"
#include <memory>
#include <optional>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <spdlog/spdlog.h>
#include "cuda.h"

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
namespace
{
std::once_flag g_cuda_watcher_atexit_once;
std::mutex g_cuda_watcher_mutex;
std::thread *g_cuda_watcher_thread = nullptr;
std::shared_ptr<std::atomic<bool>> g_cuda_watcher_stop_flag;

void stop_cuda_watcher_thread_at_exit()
{
	std::thread *thread_to_join = nullptr;
	std::shared_ptr<std::atomic<bool>> flag_to_set;
	{
		std::lock_guard<std::mutex> guard(g_cuda_watcher_mutex);
		thread_to_join = g_cuda_watcher_thread;
		flag_to_set = g_cuda_watcher_stop_flag;
		g_cuda_watcher_thread = nullptr;
		g_cuda_watcher_stop_flag.reset();
	}
	if (flag_to_set)
		flag_to_set->store(true, std::memory_order_release);
	if (thread_to_join != nullptr && thread_to_join->joinable())
		thread_to_join->join();
}
} // namespace

std::optional<attach::nv_attach_impl *>
bpf_attach_ctx::find_nv_attach_impl() const
{
	for (const auto &entry : this->attach_impl_holders) {
		if (auto p =
			    dynamic_cast<attach::nv_attach_impl *>(entry.get());
		    p)
			return p;
	}
	return {};
}
void bpf_attach_ctx::start_cuda_watcher_thread()
{
	if (cuda_watcher_thread.joinable())
		return;
	auto flag = cuda_ctx->cuda_watcher_should_stop;
	cuda_watcher_thread = std::thread([flag, this]() {
		auto *ctx = cuda_ctx.get();

		while (!flag->load()) {
			if (ctx != nullptr && ctx->cuda_shared_mem->flag1 == 1) {
				ctx->cuda_shared_mem->flag1 = 0;
				auto req_id = ctx->cuda_shared_mem->request_id;

				auto map_ptr = ctx->cuda_shared_mem->map_id;
				auto map_fd = map_ptr;
				SPDLOG_DEBUG(
					"CUDA Received call request id {}, map_ptr = {}, map_fd = {}",
					req_id, map_ptr, map_fd);
				auto start_time =
					std::chrono::high_resolution_clock::now();
				if (req_id ==
				    (int)cuda::HelperOperation::MAP_LOOKUP) {
					const auto &req =
						ctx->cuda_shared_mem->req
							.map_lookup;
					auto &resp = ctx->cuda_shared_mem->resp
							     .map_lookup;
					auto ptr = bpftime_map_lookup_elem(
						map_fd, req.key);
					resp.value = nullptr;
					if (ptr) {
						const auto host_ptr =
							reinterpret_cast<
								uintptr_t>(ptr);
						CUmemorytype mem_type =
							CU_MEMORYTYPE_HOST;
						bool is_device_ptr = false;
						if (cuPointerGetAttribute(
							    &mem_type,
							    CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
							    (CUdeviceptr)host_ptr) ==
						    CUDA_SUCCESS) {
							is_device_ptr =
								(mem_type ==
								 CU_MEMORYTYPE_DEVICE) ||
								(mem_type ==
								 CU_MEMORYTYPE_UNIFIED);
						}

						if (is_device_ptr) {
							// GPU maps can return a CUDA device pointer directly.
							resp.value = const_cast<void *>(ptr);
						} else {
							const auto comm_host_base =
								reinterpret_cast<
									uintptr_t>(
									ctx->cuda_shared_mem);
							const auto comm_device_base =
								ctx->cuda_shared_mem_device_pointer;
							const auto offset =
								static_cast<
									intptr_t>(
									host_ptr) -
								static_cast<
									intptr_t>(
									comm_host_base);
							resp.value =
								reinterpret_cast<void *>(
									static_cast<uintptr_t>(
										static_cast<intptr_t>(
											comm_device_base) +
										offset));
						}
					}
					SPDLOG_DEBUG(
						"CUDA: Executing map lookup for {}, key= {:x} result = {:x}",
						map_fd,
						*(uintptr_t *)(uintptr_t)&req
							 .key,
						(uintptr_t)resp.value);

				} else if (req_id ==
					   (int)cuda::HelperOperation::
						   MAP_UPDATE) {
					const auto &req =
						ctx->cuda_shared_mem->req
							.map_update;
					auto &resp = ctx->cuda_shared_mem->resp
							     .map_update;
					resp.result = bpftime_map_update_elem(
						map_fd, req.key, req.value,
						req.flags);
					SPDLOG_DEBUG(
						"CUDA: Executing map update for {}, result = {}",
						map_fd, resp.result);
				} else if (req_id ==
					   (int)cuda::HelperOperation::
						   MAP_DELETE) {
					const auto &req =
						ctx->cuda_shared_mem->req
							.map_delete;
					auto &resp = ctx->cuda_shared_mem->resp
							     .map_delete;

					resp.result = bpftime_map_delete_elem(
						map_fd, req.key);
					SPDLOG_DEBUG(
						"CUDA: Executing map delete for {}, result = {}",
						map_fd, resp.result);
				} else if (req_id ==
					   (int)cuda::HelperOperation::
						   TRACE_PRINTK) {
					const auto &req =
						ctx->cuda_shared_mem->req
							.trace_printk;
					auto &resp = ctx->cuda_shared_mem->resp
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
					   (int)cuda::HelperOperation::
						   GET_CURRENT_PID_TGID) {
					auto &resp = ctx->cuda_shared_mem->resp
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
					   (int)cuda::HelperOperation::PUTS) {
					const auto &req =
						ctx->cuda_shared_mem->req.puts;
					auto &resp =
						ctx->cuda_shared_mem->resp.puts;
					SPDLOG_INFO("eBPF: {}", req.data);
					resp.result = 0;
				}

				else {
					SPDLOG_WARN("Unknown request id {}",
						    req_id);
				}

				// Make sure response writes are visible before signaling completion.
				std::atomic_thread_fence(std::memory_order_seq_cst);
				ctx->cuda_shared_mem->flag2 = 1;
				std::atomic_thread_fence(
					std::memory_order_seq_cst);
			}
			std::this_thread::sleep_for(
				std::chrono::milliseconds(1));
		}
		SPDLOG_INFO("Exiting CUDA watcher thread");
	});

	std::call_once(g_cuda_watcher_atexit_once, []() {
		std::atexit(stop_cuda_watcher_thread_at_exit);
	});
	{
		std::lock_guard<std::mutex> guard(g_cuda_watcher_mutex);
		g_cuda_watcher_thread = &cuda_watcher_thread;
		g_cuda_watcher_stop_flag = flag;
	}
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
		entry.map_type = 0;
		entry.extra_buffer = nullptr;
		entry.max_thread_count = 0;
	}
	const auto &handler_manager =
		*shm_holder.global_shared_memory.get_manager();
	for (size_t i = 0; i < handler_manager.size(); i++) {
		const auto &current_handler = handler_manager.get_handler(i);
		if (std::holds_alternative<bpf_map_handler>(current_handler)) {
			auto &local = local_basic_info[i];
			const auto &map =
				std::get<bpf_map_handler>(current_handler);
			auto gpu_buffer = map.get_gpu_map_extra_buffer();
			SPDLOG_INFO(
				"Copying map fd {} to device, key size={}, value size={}, max ent={}, map_type = {}, gpu_buffer = {:x}",
				i, map.get_key_size(), map.get_value_size(),
				map.get_max_entries(), (int)map.type,
				(uintptr_t)gpu_buffer);

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
			local.map_type = (int)map.type;
			local.extra_buffer = gpu_buffer;
			local.max_thread_count =
				map.get_gpu_map_max_thread_count();
		}
	}

	return local_basic_info;
}

namespace cuda
{

void cuda_context_destroyer(CUcontext ptr)
{
	NV_SAFE_CALL(cuCtxDestroy(ptr), "destroy cuda context");
}
void cuda_module_destroyer(CUmodule ptr)
{
	NV_SAFE_CALL(cuModuleUnload(ptr), "Unload CUDA module");
}

std::optional<std::unique_ptr<cuda::CUDAContext>> create_cuda_context()
{
	SPDLOG_INFO("Initializing CUDA shared memory");
	auto *cuda_shared_mem =
		shm_holder.global_shared_memory.get_cuda_comm_shared_mem();
	if (!cuda_shared_mem) {
		SPDLOG_ERROR(
			"CUDA shared communication memory not initialized in shared segment");
		return std::nullopt;
	}
	memset(cuda_shared_mem, 0, sizeof(*cuda_shared_mem));

	auto cuda_ctx = std::make_optional(
		std::make_unique<cuda::CUDAContext>(cuda_shared_mem));

	SPDLOG_INFO("CUDA context created");
	return cuda_ctx;
}
CUDAContext::~CUDAContext()
{
	SPDLOG_INFO("Destructing CUDAContext");
}
CUDAContext::CUDAContext(cuda::CommSharedMem *mem)
	: cuda_shared_mem(mem), cuda_shared_mem_device_pointer(0)

{
	// Move CommSharedMem from the agent’s local memory to shared memory to
	// improve performance.
	void *device_ptr = nullptr;
	auto err = cudaHostGetDevicePointer(&device_ptr,
					    (void *)cuda_shared_mem, 0);
	if (err != cudaSuccess) {
		SPDLOG_ERROR(
			"cudaHostGetDevicePointer failed for CommSharedMem: {}",
			cudaGetErrorString(err));
		throw std::runtime_error(
			"Unable to get device pointer for CommSharedMem");
	}
	cuda_shared_mem_device_pointer =
		reinterpret_cast<uintptr_t>(device_ptr);
	set_cuda_shared_mem_device_pointer(cuda_shared_mem_device_pointer);
	SPDLOG_INFO("CommSharedMem host {:p} mapped to device {:p}",
		    (void *)cuda_shared_mem, device_ptr);
}

} // namespace cuda
} // namespace bpftime
