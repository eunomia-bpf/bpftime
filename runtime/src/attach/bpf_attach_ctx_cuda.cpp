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
#include <pthread.h>
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

// Implement CUDAContext methods in cuda namespace
namespace cuda
{
void CUDAContext::CudaHostMemoryDeleter::operator()(CommSharedMem *ptr) const
{
	if (ptr) {
		auto err = cudaFreeHost(ptr);
		if (err != cudaSuccess) {
			SPDLOG_ERROR(
				"Failed to free host memory with cudaFreeHost {}",
				cudaGetErrorString(err));
		}
	}
}

CUDAContext::CUDAContext(CommSharedMemPtr &&mem)
	: cuda_shared_mem(std::move(mem)),
	  should_stop(false)
{
	worker_thread = std::thread(&CUDAContext::worker_func, this);
}

CUDAContext::~CUDAContext()
{
	SPDLOG_INFO("Destructing CUDAContext");
	should_stop.store(true);
	if (worker_thread.joinable()) {
		worker_thread.join();
	}
	// cudaHostAlloc allocated memory doesn't need cudaHostUnregister
}

void CUDAContext::worker_func()
{
#ifdef __linux__
	// Set thread name in the executing thread context (max 15 chars + null)
	int setname_ret = pthread_setname_np(pthread_self(), "cuda_watcher");
	if (setname_ret != 0) {
		SPDLOG_ERROR("Failed to set thread name: pthread_setname_np returned {}", setname_ret);
	}
#endif

	SPDLOG_INFO("CUDA watcher thread started");

	while (!should_stop.load()) {
		// use volatile to ensure that each time reads from memory
		volatile int flag1_value = cuda_shared_mem->flag1;
		if (flag1_value == 1) {
			cuda_shared_mem->flag1 = 0;
#ifdef __aarch64__
			std::atomic_thread_fence(
				std::memory_order_acquire); // arm is weak memory order
#endif
			auto req_id = cuda_shared_mem->request_id;

			auto map_ptr = cuda_shared_mem->map_id;
			auto map_fd = map_ptr;
			SPDLOG_DEBUG(
				"CUDA Received call request id {}, map_ptr = {}, map_fd = {}",
				req_id, map_ptr, map_fd);

			if (req_id == (int)HelperOperation::MAP_LOOKUP) {
				const auto &req =
					cuda_shared_mem->req.map_lookup;
				auto &resp = cuda_shared_mem->resp.map_lookup;
				auto ptr = bpftime_map_lookup_elem(map_fd,
								   req.key);
				resp.value = ptr;
				SPDLOG_DEBUG(
					"CUDA: Executing map lookup for {}, key= {:x} result = {:x}",
					map_fd,
					*(uintptr_t *)(uintptr_t)&req.key,
					(uintptr_t)resp.value);

			} else if (req_id == (int)HelperOperation::MAP_UPDATE) {
				const auto &req =
					cuda_shared_mem->req.map_update;
				auto &resp = cuda_shared_mem->resp.map_update;
				resp.result = bpftime_map_update_elem(
					map_fd, req.key, req.value, req.flags);
				SPDLOG_DEBUG(
					"CUDA: Executing map update for {}, result = {}",
					map_fd, resp.result);
			} else if (req_id == (int)HelperOperation::MAP_DELETE) {
				const auto &req =
					cuda_shared_mem->req.map_delete;
				auto &resp = cuda_shared_mem->resp.map_delete;

				resp.result = bpftime_map_delete_elem(map_fd,
								      req.key);
				SPDLOG_DEBUG(
					"CUDA: Executing map delete for {}, result = {}",
					map_fd, resp.result);
			} else if (req_id ==
				   (int)HelperOperation::TRACE_PRINTK) {
				const auto &req =
					cuda_shared_mem->req.trace_printk;
				auto &resp = cuda_shared_mem->resp.trace_printk;

				resp.result = bpftime_trace_printk(
					(uintptr_t)req.fmt, req.fmt_size,
					req.arg1, req.arg2, req.arg3);
				SPDLOG_DEBUG(
					"CUDA: Executing bpf_printk: {}, arg1={}, arg2={}, arg3={}",
					req.fmt, req.arg1, req.arg2, req.arg3);
			} else if (req_id ==
				   (int)HelperOperation::GET_CURRENT_PID_TGID) {
				auto &resp = cuda_shared_mem->resp.get_tid_pgid;
				static int tgid = getpid();
				static thread_local int tid = -1;
				if (tid == -1) {
					tid = gettid();
				}
				SPDLOG_DEBUG(
					"Called get_current_pid_tgid: pid={}, tgid={}",
					tid, tgid);
				resp.result = (((uint64_t)tgid) << 32) | tid;
			} else if (req_id == (int)HelperOperation::PUTS) {
				const auto &req = cuda_shared_mem->req.puts;
				auto &resp = cuda_shared_mem->resp.puts;
				SPDLOG_INFO("eBPF: {}", req.data);
				resp.result = 0;
			} else {
				SPDLOG_WARN("Unknown request id {}", req_id);
			}

			cuda_shared_mem->flag2 = 1;
			std::atomic_thread_fence(std::memory_order_seq_cst);
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	SPDLOG_INFO("Exiting CUDA watcher thread");
}

void cuda_context_destroyer(CUcontext ptr)
{
	NV_SAFE_CALL(cuCtxDestroy(ptr), "destroy cuda context");
}
void cuda_module_destroyer(CUmodule ptr)
{
	NV_SAFE_CALL(cuModuleUnload(ptr), "Unload CUDA module");
}

std::optional<std::unique_ptr<CUDAContext>> create_cuda_context()
{
	SPDLOG_INFO("Initializing CUDA shared memory");

	CommSharedMem *ptr = nullptr;
	auto err = cudaHostAlloc((void **)&ptr, sizeof(CommSharedMem),
				 cudaHostAllocDefault);
	if (err != cudaSuccess) {
		SPDLOG_ERROR("cudaHostAlloc failed: {} ({})",
			     cudaGetErrorName(err), cudaGetErrorString(err));
		return std::nullopt;
	}

	CUDAContext::CommSharedMemPtr cuda_shared_mem(ptr);

	SPDLOG_INFO("Allocated {} MB of pinned memory using cudaHostAlloc",
		sizeof(CommSharedMem) / (1024.0 * 1024));

	memset(cuda_shared_mem.get(), 0, sizeof(CommSharedMem));

	SPDLOG_INFO("Successfully allocated pinned memory via cudaHostAlloc");

	auto cuda_ctx = std::make_optional(
		std::make_unique<CUDAContext>(std::move(cuda_shared_mem)));

	SPDLOG_INFO("CUDA context created");
	return cuda_ctx;
}

} // namespace cuda

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

} // namespace bpftime
