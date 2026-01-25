#include "nv_gpu_shared_array_map.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include "nv_gpu_gdrcopy.hpp"
#include "cuda.h"
#include "linux/bpf.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>
#include <unistd.h>

using namespace bpftime;

nv_gpu_shared_array_map_impl::nv_gpu_shared_array_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint64_t value_size,
	uint64_t max_entries)
	: agent_cuda_context(nullptr), agent_stream(nullptr),
	  agent_gpu_shared_mem(memory.get_segment_manager()),
	  value_size(value_size), max_entries(max_entries),
	  value_buffer(memory.get_segment_manager())
{
	// Record the CUDA context that owns the device memory on server side
	owner_cuda_context = nullptr;
	if (shm_holder.global_shared_memory.get_open_type() !=
	    shm_open_type::SHM_OPEN_ONLY) {
		cuCtxGetCurrent(&owner_cuda_context);
		if (!owner_cuda_context) {
			SPDLOG_WARN(
				"Owner CUDA context not set at init; CUDA calls may fail");
		}
	}
	value_buffer.resize(value_size);
	auto total_buffer_size = (uint64_t)value_size * max_entries;
	SPDLOG_INFO(
		"Initializing map type of BPF_MAP_TYPE_GPU_ARRAY_MAP (device), total_buffer_size={}",
		total_buffer_size);
	if (auto err = cuMemAlloc(&server_gpu_shared_mem, total_buffer_size);
	    err != CUDA_SUCCESS) {
		SPDLOG_ERROR(
			"Unable to allocate GPU buffer for nv_gpu_shared_array_map_impl: {}",
			(int)err);
		throw std::runtime_error(
			"Unable to allocate GPU buffer for nv_gpu_shared_array_map_impl");
	}
	if (auto err = cuMemsetD8(server_gpu_shared_mem, 0, total_buffer_size);
	    err != CUDA_SUCCESS) {
		SPDLOG_ERROR("Unable to fill GPU buffer with zero: {}",
			     (int)err);
	}
	if (auto err = cuIpcGetMemHandle(&this->gpu_mem_handle,
					 server_gpu_shared_mem);
	    err != CUDA_SUCCESS) {
		SPDLOG_ERROR(
			"Unable to open CUDA IPC handle for nv_gpu_shared_array_map_impl: {}",
			(int)err);
		throw std::runtime_error(
			"Unable to open CUDA IPC handle for nv_gpu_shared_array_map_impl");
	}
}

void *nv_gpu_shared_array_map_impl::elem_lookup(const void *key)
{
	auto key_val = *(uint32_t *)key;
	if (key_val >= max_entries) {
		errno = ENOENT;
		return nullptr;
	}
	auto base = try_initialize_for_agent_and_get_mapped_address();
	if (shm_holder.global_shared_memory.get_open_type() ==
		    shm_open_type::SHM_OPEN_ONLY &&
	    (base == 0 || agent_cuda_context == nullptr)) {
		errno = EINVAL;
		return nullptr;
	}
	auto total_buffer_size = (uint64_t)value_size * max_entries;
	auto copy_offset_bytes = (uint64_t)key_val * value_size;
	if (bpftime::gpu::gdrcopy::copy_from_device_to_host_with_gdrcopy(
		    this, "BPF_MAP_TYPE_GPU_ARRAY_MAP", base, total_buffer_size,
		    copy_offset_bytes, value_buffer.data(), value_size,
		    value_size)) {
		return value_buffer.data();
	}
	CUcontext target_ctx = nullptr;
	if (shm_holder.global_shared_memory.get_open_type() ==
	    shm_open_type::SHM_OPEN_ONLY) {
		target_ctx = agent_cuda_context;
	} else {
		target_ctx = owner_cuda_context;
	}
	cuCtxPushCurrent_v2(target_ctx);

	CUresult err = CUDA_SUCCESS;

	if (shm_holder.global_shared_memory.get_open_type() ==
	    shm_open_type::SHM_OPEN_ONLY) {
		cuMemcpyDtoHAsync_v2(
			value_buffer.data(),
			(CUdeviceptr)base + (uint64_t)key_val * value_size,
			value_size, agent_stream);
		err = cuStreamSynchronize(agent_stream);
	} else {
		err = cuMemcpyDtoH(value_buffer.data(),
				   (CUdeviceptr)base +
					   (uint64_t)key_val * value_size,
				   value_size);
	}

	if (err != CUDA_SUCCESS) {
		SPDLOG_ERROR(
			"Unable to copy bytes from GPU to host with stream {}: {}",
			(intptr_t)agent_stream, (int)err);
		cuCtxPopCurrent(&target_ctx);
		return nullptr;
	}
	cuCtxPopCurrent(&target_ctx);
	SPDLOG_DEBUG("Copied GPU memory base {:x} offset {} size {} to host",
		     base, (uint64_t)key_val * value_size, value_size);
	return value_buffer.data();
}

long nv_gpu_shared_array_map_impl::elem_update(const void *key,
					       const void *value,
					       uint64_t flags)
{
	if (!bpftime::check_update_flags(flags))
		return -1;
	auto key_val = *(uint32_t *)key;
	if ((key_val < max_entries) && flags == BPF_NOEXIST) {
		errno = EEXIST;
		return -1;
	}
	if (key_val >= max_entries) {
		errno = E2BIG;
		return -1;
	}
	auto base = try_initialize_for_agent_and_get_mapped_address();
	if (shm_holder.global_shared_memory.get_open_type() ==
		    shm_open_type::SHM_OPEN_ONLY &&
	    (base == 0 || agent_cuda_context == nullptr)) {
		errno = EINVAL;
		return -1;
	}
	CUcontext target_ctx = nullptr;
	if (shm_holder.global_shared_memory.get_open_type() ==
	    shm_open_type::SHM_OPEN_ONLY) {
		target_ctx = agent_cuda_context;
	} else {
		target_ctx = owner_cuda_context;
	}

	cuCtxPushCurrent_v2(target_ctx);

	CUresult err = CUDA_SUCCESS;
	if (shm_holder.global_shared_memory.get_open_type() ==
	    shm_open_type::SHM_OPEN_ONLY) {
		cuMemcpyHtoDAsync_v2((CUdeviceptr)base +
					     (uint64_t)key_val * value_size,
				     value, value_size, agent_stream);
		err = cuStreamSynchronize(agent_stream);
	} else {
		err = cuMemcpyHtoD((CUdeviceptr)base +
					   (uint64_t)key_val * value_size,
				   value, value_size);
	}

	if (err != CUDA_SUCCESS) {
		SPDLOG_ERROR(
			"Unable to copy {} bytes from host to GPU with stream {}: {}",
			value_size, (intptr_t)agent_stream, (int)err);
		errno = EINVAL;
		cuCtxPopCurrent(&target_ctx);
		return -1;
	}
	cuCtxPopCurrent(&target_ctx);
	return 0;
}

long nv_gpu_shared_array_map_impl::elem_delete(const void *key)
{
	errno = EINVAL;
	return -1;
}

int nv_gpu_shared_array_map_impl::map_get_next_key(const void *key,
						   void *next_key)
{
	auto &next_key_val = *(uint32_t *)next_key;

	if (key == nullptr) {
		next_key_val = 0;
		return 0;
	} else {
		auto key_val = *(uint32_t *)key;
		if (key_val >= max_entries) {
			next_key = 0;
			return 0;
		}
		if (key_val + 1 == max_entries) {
			errno = ENOENT;
			return -1;
		}
		next_key_val = key_val + 1;
		return 0;
	}
}

nv_gpu_shared_array_map_impl::~nv_gpu_shared_array_map_impl()
{
	auto total_buffer_size = (uint64_t)value_size * max_entries;
	bpftime::gpu::gdrcopy::destroy_gdrcopy_mapping_for_owner(
		this, "BPF_MAP_TYPE_GPU_ARRAY_MAP", total_buffer_size);
	if (shm_holder.global_shared_memory.get_open_type() !=
	    shm_open_type::SHM_OPEN_ONLY) {
		// Server side: free device memory
		CUcontext prev_ctx = nullptr;
		bool did_switch_ctx = false;
		if (owner_cuda_context) {
			cuCtxGetCurrent(&prev_ctx);
			if (prev_ctx != owner_cuda_context) {
				cuCtxSetCurrent(owner_cuda_context);
				did_switch_ctx = true;
			}
		}
		if (auto err = cuMemFree(server_gpu_shared_mem);
		    err != CUDA_SUCCESS) {
			SPDLOG_WARN(
				"Unable to free CUDA memory for nv_gpu_shared_array_map_impl: {}",
				(int)err);
		}
		if (did_switch_ctx) {
			cuCtxSetCurrent(prev_ctx);
		}
	} else {
		// Agent side: nothing allocated here; mapping closed by CUDA
		// context
		if (agent_stream) {
			cuStreamDestroy(agent_stream);
		}
	}
}

CUdeviceptr
nv_gpu_shared_array_map_impl::try_initialize_for_agent_and_get_mapped_address()
{
	if (shm_holder.global_shared_memory.get_open_type() ==
	    shm_open_type::SHM_OPEN_ONLY) {
		int pid = getpid();
		if (auto itr = agent_gpu_shared_mem.find(pid);
		    itr == agent_gpu_shared_mem.end()) {
			SPDLOG_INFO(
				"Initializing nv_gpu_shared_array_map_impl at pid {}",
				pid);
			if (auto err = cuInit(0); err != CUDA_SUCCESS) {
				SPDLOG_ERROR("cuInit failed: {}", (int)err);
				errno = EINVAL;
				return 0;
			}

			// Build candidate device indices.
			std::vector<int> candidates;
			if (const char *p = getenv("BPFTIME_CUDA_DEVICE");
			    p && p[0] != '\0') {
				errno = 0;
				char *end = nullptr;
				long v = strtol(p, &end, 10);
				if (errno == 0 && end && end[0] == '\0' && v >= 0 &&
				    v <= std::numeric_limits<int>::max()) {
					candidates.push_back((int)v);
				}
			}
			int count = 0;
			if (cuDeviceGetCount(&count) == CUDA_SUCCESS && count > 0) {
				for (int i = 0; i < count; i++)
					candidates.push_back(i);
			} else {
				candidates.push_back(0);
			}
			std::sort(candidates.begin(), candidates.end());
			candidates.erase(
				std::unique(candidates.begin(), candidates.end()),
				candidates.end());

			// Try current context first if available.
			CUdeviceptr ptr = 0;
			CUcontext prev_ctx = nullptr;
			cuCtxGetCurrent(&prev_ctx);
			if (prev_ctx != nullptr) {
				auto err2 = cuIpcOpenMemHandle(
					&ptr, gpu_mem_handle,
					CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
				if (err2 == CUDA_SUCCESS) {
					agent_cuda_context = prev_ctx;
				} else {
					ptr = 0;
				}
			}

			// Otherwise probe devices by temporarily setting their primary ctx.
			if (ptr == 0) {
				for (int d : candidates) {
					CUdevice dev = 0;
					if (cuDeviceGet(&dev, d) != CUDA_SUCCESS)
						continue;
					CUcontext ctx = nullptr;
					if (cuDevicePrimaryCtxRetain(&ctx, dev) !=
						    CUDA_SUCCESS ||
					    ctx == nullptr)
						continue;
					if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
						cuDevicePrimaryCtxRelease(dev);
						continue;
					}
					auto err2 = cuIpcOpenMemHandle(
						&ptr, gpu_mem_handle,
						CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
					if (err2 == CUDA_SUCCESS && ptr != 0) {
						agent_cuda_context = ctx;
						SPDLOG_INFO(
							"Mapped CUDA IPC handle on device {}",
							d);
						break;
					}
					ptr = 0;
					cuDevicePrimaryCtxRelease(dev);
				}
			}

			// Restore previous context for this thread.
			if (prev_ctx != nullptr)
				cuCtxSetCurrent(prev_ctx);

			if (ptr == 0 || agent_cuda_context == nullptr) {
				SPDLOG_ERROR(
					"Unable to map CUDA IPC memory for shared array map on any device");
				errno = EINVAL;
				return 0;
			}

			// Create a private stream in the chosen agent context.
			if (agent_stream == nullptr) {
				CUcontext tmp_prev = nullptr;
				cuCtxGetCurrent(&tmp_prev);
				if (tmp_prev != agent_cuda_context)
					cuCtxSetCurrent(agent_cuda_context);
				CUresult err2 = cuStreamCreate(&agent_stream,
							       CU_STREAM_NON_BLOCKING);
				if (err2 != CUDA_SUCCESS || agent_stream == nullptr) {
					SPDLOG_ERROR(
						"Unable to create CUDA stream for agent: {}",
						(int)err2);
					if (tmp_prev != agent_cuda_context)
						cuCtxSetCurrent(tmp_prev);
					errno = EINVAL;
					return 0;
				}
				if (tmp_prev != agent_cuda_context)
					cuCtxSetCurrent(tmp_prev);
			}

			SPDLOG_INFO(
				"Mapped GPU memory for shared array map: {}",
				(uintptr_t)ptr);
			agent_gpu_shared_mem[pid] = ptr;
		}
		return agent_gpu_shared_mem[pid];
	} else {
		return server_gpu_shared_mem;
	}
}
