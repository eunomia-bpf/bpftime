#include "nv_gpu_shared_array_map.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include "cuda.h"
#include "linux/bpf.h"
#include "spdlog/spdlog.h"
#include <cerrno>
#include <cstdint>
#include <stdexcept>
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
		cuMemcpyDtoHAsync_v2(value_buffer.data(),
				     (CUdeviceptr)base +
					     (uint64_t)key_val * value_size,
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
		if (agent_cuda_context == nullptr) {
			cuCtxGetCurrent(&agent_cuda_context);
			if (agent_cuda_context == nullptr) {
				SPDLOG_ERROR(
					"Unable to get CUDA context for agent");
				throw std::runtime_error(
					"Unable to get CUDA context for agent");
			}
			SPDLOG_DEBUG(
				"try_initialize_for_agent_and_get_mapped_address with context: {}",
				(uintptr_t)agent_cuda_context);
		}
		if (agent_stream == nullptr) {
			auto err = cuStreamCreate(&agent_stream,
						  CU_STREAM_NON_BLOCKING);
			if (agent_stream == nullptr || err != CUDA_SUCCESS) {
				SPDLOG_ERROR(
					"Unable to create CUDA stream for agent");
				throw std::runtime_error(
					"Unable to create CUDA stream for agent");
			}
			SPDLOG_DEBUG(
				"try_initialize_for_agent_and_get_mapped_address creates stream: {}",
				(uintptr_t)agent_stream);
		}

		int pid = getpid();
		if (auto itr = agent_gpu_shared_mem.find(pid);
		    itr == agent_gpu_shared_mem.end()) {
			SPDLOG_INFO(
				"Initializing nv_gpu_shared_array_map_impl at pid {}",
				pid);
			CUdeviceptr ptr;
			if (auto err = cuIpcOpenMemHandle(
				    &ptr, gpu_mem_handle,
				    CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
			    err != CUDA_SUCCESS) {
				SPDLOG_ERROR(
					"Unable to map CUDA IPC memory for shared array map, error={}",
					(int)err);
				throw std::runtime_error(
					"Unable to map CUDA IPC memory for shared array map!");
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
