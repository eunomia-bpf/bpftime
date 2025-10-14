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
#include <spdlog/spdlog.h>
#include "cuda.h"
#include <cstring>

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
void bpf_attach_ctx::start_cuda_watcher_thread()
{
	auto flag = this->cuda_ctx->cuda_watcher_should_stop;
	std::thread handle([=, this]() {
		SPDLOG_INFO("CUDA watcher thread started");
		auto &ctx = cuda_ctx;

		while (!flag->load()) {
			if (ctx->cuda_shared_mem->flag1 == 1) {
				ctx->cuda_shared_mem->flag1 = 0;
				auto req_id = ctx->cuda_shared_mem->request_id;

				long raw_map = ctx->cuda_shared_mem->map_id;
				unsigned long long raw_u =
					(unsigned long long)raw_map;
				int map_fd = (int)(raw_u >> 32);
				if (map_fd <= 0 || map_fd > 1 << 30) {
					map_fd = (int)(raw_u & 0xffffffffull);
				}
				// Validate/translate map_fd: fallback helper
				// (applied for all requests) Rationale:
				// device-side map-id encoding can sometimes be
				// invalid or noisy. Normalizing fd here makes
				// subsequent handling more robust at the cost
				// of potentially more frequent fallback logs
				// across helper types.
				auto ensure_valid_gpu_array_fd =
					[&](int fd) -> int {
					const auto *manager =
						shm_holder.global_shared_memory
							.get_manager();
					if (!manager)
						return fd;
					// Accept only GPU_ARRAY_MAP for
					// implicit fallback. Other map types
					// are not zero-copy GPU shared memory
					// maps and should not be implicitly
					// translated.
					auto is_valid_gpu_array =
						[&](int x) -> bool {
						if (x < 0 ||
						    x >= (int)manager->size() ||
						    !manager->is_allocated(x))
							return false;
						const auto &h =
							manager->get_handler(x);
						if (!std::holds_alternative<
							    bpf_map_handler>(h))
							return false;
						const auto &mh = std::get<
							bpf_map_handler>(h);
						return mh.type ==
						       bpf_map_type::
							       BPF_MAP_TYPE_GPU_ARRAY_MAP;
					};
					if (is_valid_gpu_array(fd))
						return fd;
					int target = -1;
					for (int i = 0;
					     i < (int)manager->size(); i++) {
						if (!is_valid_gpu_array(i))
							continue;
						target = i;
						break;
					}
					if (target != -1) {
						SPDLOG_INFO(
							"Fallback map fd from {} to {} (GPU_ARRAY_MAP)",
							fd, target);
						return target;
					}
					return fd;
				};
				// Apply fd fallback for all incoming requests
				// Rationale: normalize invalid/placeholder fds
				// before any branch-specific logic.
				map_fd = ensure_valid_gpu_array_fd(map_fd);
				SPDLOG_INFO(
					"CUDA Received call request id {}, raw_map = {}, decoded map_fd = {}",
					req_id, raw_map, map_fd);
				if (req_id ==
				    (int)cuda::HelperOperation::MAP_LOOKUP) {
					const auto &req =
						ctx->cuda_shared_mem->req
							.map_lookup;
					auto &resp = ctx->cuda_shared_mem->resp
							     .map_lookup;
					// For GPU_ARRAY_MAP: copy key into a
					// local buffer. Rationale:
					// CommSharedMem request buffer can be
					// reused by other helpers; local copy
					// avoids key contamination (previously
					// observed as ASCII remnants).
					int map_type_for_lookup = -1;
					if (const auto *manager =
						    shm_holder
							    .global_shared_memory
							    .get_manager()) {
						if (map_fd >= 0 &&
						    map_fd < (int)manager
								     ->size() &&
						    manager->is_allocated(
							    map_fd)) {
							const auto &h =
								manager->get_handler(
									map_fd);
							if (std::holds_alternative<
								    bpf_map_handler>(
								    h))
								map_type_for_lookup =
									(int)std::get<
										bpf_map_handler>(
										h)
										.type;
						}
					}
					// Copy key to a local buffer to avoid
					// shared-memory reuse contamination
					uint32_t key_local_l = 0;
					const void *lookup_key_ptr =
						(const void *)&req.key;
					if (map_type_for_lookup ==
					    (int)bpf_map_type::
						    BPF_MAP_TYPE_GPU_ARRAY_MAP) {
						std::memcpy(
							&key_local_l,
							(const void *)&req.key,
							sizeof(uint32_t));
						lookup_key_ptr =
							(const void
								 *)&key_local_l;
					}
					auto ptr = bpftime_map_lookup_elem(
						map_fd, lookup_key_ptr);
					// Copy looked-up value into
					// shared-memory value slot and return a
					// device-visible pointer.
					// Rationale: device must dereference a
					// device-visible address; we compute it
					// as device_base + (host_value_addr -
					// host_base) after host memcpy.
					if (ptr) {
						uint32_t value_size =
							bpftime_map_value_size_from_syscall(
								map_fd);
						std::memcpy(
							(void *)&ctx
								->cuda_shared_mem
								->req.map_update
								.value,
							ptr, value_size);
						uintptr_t host_base =
							(uintptr_t)ctx
								->cuda_shared_mem
								.get();
						uintptr_t host_value_addr =
							(uintptr_t)&ctx
								->cuda_shared_mem
								->req.map_update
								.value;
						uintptr_t offset =
							host_value_addr -
							host_base;
						uintptr_t dev_base =
							ctx->cuda_shared_mem_device_pointer;
						resp.value =
							(const void *)(dev_base +
								       offset);
					} else {
						resp.value = nullptr;
					}
					SPDLOG_DEBUG(
						"CUDA: map lookup for {}, resp.value(dev)={:x}",
						map_fd, (uintptr_t)resp.value);

				} else if (req_id ==
					   (int)cuda::HelperOperation::
						   MAP_UPDATE) {
					const auto &req =
						ctx->cuda_shared_mem->req
							.map_update;
					auto &resp = ctx->cuda_shared_mem->resp
							     .map_update;
					// Identify map type and sizes for
					// correct handling
					int map_type_for_update = -1;
					uint32_t m_key_size_u = 0,
						 m_value_size_u = 0;
					if (const auto *manager =
						    shm_holder
							    .global_shared_memory
							    .get_manager()) {
						if (map_fd >= 0 &&
						    map_fd < (int)manager
								     ->size() &&
						    manager->is_allocated(
							    map_fd)) {
							const auto &h =
								manager->get_handler(
									map_fd);
							if (std::holds_alternative<
								    bpf_map_handler>(
								    h)) {
								const auto &mh = std::get<
									bpf_map_handler>(
									h);
								map_type_for_update =
									(int)mh.type;
								m_key_size_u =
									mh.get_key_size();
								m_value_size_u =
									mh.get_value_size();
							}
						}
					}
					// Copy key to a local buffer to avoid
					// shared-memory reuse contamination
					uint32_t key_local_u = 0;
					const void *update_key_ptr =
						(const void *)&req.key;
					if (map_type_for_update ==
					    (int)bpf_map_type::
						    BPF_MAP_TYPE_GPU_ARRAY_MAP) {
						std::memcpy(
							&key_local_u,
							(const void *)&req.key,
							sizeof(uint32_t));
						update_key_ptr =
							(const void
								 *)&key_local_u;
					}
					resp.result = bpftime_map_update_elem(
						map_fd, update_key_ptr,
						req.value, req.flags);
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

				ctx->cuda_shared_mem->flag2 = 1;
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
			void *gpu_buffer_dev = (void *)gpu_buffer;
#ifdef BPFTIME_ENABLE_CUDA_ATTACH
			// Translate host pointer to a device-visible pointer
			// (for host-shared GPU maps)
			if (map.type ==
				    bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_MAP ||
			    map.type ==
				    bpf_map_type::BPF_MAP_TYPE_GPU_RINGBUF_MAP) {
				void *tmp = nullptr;
				auto err = cudaHostGetDevicePointer(
					&tmp, gpu_buffer_dev, 0);
				if (err == cudaSuccess && tmp != nullptr) {
					gpu_buffer_dev = tmp;
				} else {
					SPDLOG_WARN(
						"cudaHostGetDevicePointer failed for map fd {}: {} (fallback to host ptr)",
						i, (int)err);
				}
			}
#endif
			SPDLOG_INFO(
				"Copying map fd {} to device, key size={}, value size={}, max ent={}, map_type = {}, gpu_buffer = {:x}",
				i, map.get_key_size(), map.get_value_size(),
				map.get_max_entries(), (int)map.type,
				(uintptr_t)gpu_buffer_dev);

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
			local.extra_buffer = gpu_buffer_dev;
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
	auto cuda_shared_mem = std::make_unique<cuda::CommSharedMem>();
	memset(cuda_shared_mem.get(), 0, sizeof(*cuda_shared_mem));

	CUDART_SAFE_CALL(cudaHostRegister(cuda_shared_mem.get(),
					  sizeof(cuda::CommSharedMem),
					  cudaHostRegisterMapped),
			 "Unable to register shared memory");

	auto cuda_ctx = std::make_optional(std::make_unique<cuda::CUDAContext>(
		std::move(cuda_shared_mem)));
	// Compute device-visible base address of shared memory for building
	// device-visible resp.value pointers
	{
		void *device_ptr = nullptr;
		auto err = cudaHostGetDevicePointer(
			&device_ptr, (void *)(*cuda_ctx)->cuda_shared_mem.get(),
			0);
		if (err == cudaSuccess && device_ptr != nullptr) {
			(*cuda_ctx)->cuda_shared_mem_device_pointer =
				reinterpret_cast<uintptr_t>(device_ptr);
			SPDLOG_INFO(
				"Mapped CommSharedMem to device pointer {:x}",
				(uintptr_t)device_ptr);
		} else {
			SPDLOG_WARN(
				"cudaHostGetDevicePointer failed for CommSharedMem: {}",
				(int)err);
		}
	}

	SPDLOG_INFO("CUDA context created");
	return cuda_ctx;
}
CUDAContext::~CUDAContext()
{
	SPDLOG_INFO("Destructing CUDAContext");
	if (auto result = cudaHostUnregister(cuda_shared_mem.get());
	    result != cudaSuccess) {
		SPDLOG_ERROR("Unable to unregister host memory: {}",
			     (int)result);
	}
}
CUDAContext::CUDAContext(std::unique_ptr<cuda::CommSharedMem> &&mem)
	: cuda_shared_mem(std::move(mem)),
	  cuda_shared_mem_device_pointer((uintptr_t)cuda_shared_mem.get())

{
}

} // namespace cuda
} // namespace bpftime