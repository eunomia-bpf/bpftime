#include "nv_gpu_ringbuf_map.hpp"
#include "bpftime_shm_internal.hpp"
#include "cuda.h"
#include "spdlog/spdlog.h"
#include <cstring>
#include <stdexcept>
using namespace bpftime;

nv_gpu_ringbuf_map_impl::nv_gpu_ringbuf_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint64_t value_size,
	uint64_t max_entries, uint64_t thread_count)
	: server_shared_mem(memory.get_segment_manager()),
	  agent_gpu_shared_mem(memory.get_segment_manager()),
	  value_size(value_size), max_entries(max_entries),
	  thread_count(thread_count), local_buffer(memory.get_segment_manager())
{
	entry_size = value_size * max_entries + sizeof(ringbuf_header);
	local_buffer.resize(entry_size);
	if (auto err = cuMemAllocManaged((CUdeviceptr *)&server_shared_mem,
					 thread_count * entry_size,
					 CU_MEM_ATTACH_GLOBAL);
	    err != CUDA_SUCCESS) {
		SPDLOG_ERROR("Unable to allocate host-device shared memory: {}",
			     (int)err);
		throw std::runtime_error(
			"Unable to allocate host-device shared memory");
	}
	if (auto err = cuIpcGetMemHandle(&gpu_mem_handle,
					 (CUdeviceptr)server_shared_mem);
	    err != CUDA_SUCCESS) {
		SPDLOG_ERROR("Unable to register gpu memory to IPC: {}",
			     (int)err);
		throw std::runtime_error(
			"Unable to register gpu memory to IPC");
	}
}

CUdeviceptr
nv_gpu_ringbuf_map_impl::try_initialize_for_agent_and_get_mapped_address()
{
	if (shm_holder.global_shared_memory.get_open_type() !=
	    shm_open_type::SHM_CREATE_OR_OPEN) {
		int pid = getpid();
		if (auto itr = agent_gpu_shared_mem.find(pid);
		    itr == agent_gpu_shared_mem.end()) {
			SPDLOG_INFO(
				"Initializing nv_gpu_array_map_impl at pid {}",
				pid);
			CUdeviceptr ptr;
			if (auto err = cuIpcOpenMemHandle(
				    &ptr, gpu_mem_handle,
				    CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
			    err != CUDA_SUCCESS) {
				SPDLOG_ERROR(
					"Unable to map CUDA IPC memory, error={}",
					(int)err);
				throw std::runtime_error(
					"Unable to map CUDA IPC memory!");
			}
			SPDLOG_INFO("Mapped GPU memory for gpu array map: {}",
				    ptr);
			agent_gpu_shared_mem[pid] = ptr;
		}
		return agent_gpu_shared_mem[pid];
	} else {
		return (CUdeviceptr)server_shared_mem;
	}
}

int nv_gpu_ringbuf_map_impl::drain_data(
	const std::function<void(const void *)> &fn)
{
	for (uint64_t i = 0; i < thread_count; i++) {
		auto header =
			(ringbuf_header *)(uintptr_t)(i * entry_size +
						      (char *)server_shared_mem);
		if (header->head != header->tail) {
			// Got data!
			auto real_head = header->head % max_entries;
			// Copy data to local buffer
			memcpy(local_buffer.data(),
			       (char *)header + real_head * value_size,
			       value_size);
			__atomic_add_fetch(&header->head, 1, __ATOMIC_SEQ_CST);
			fn(local_buffer.data());
		}
	}

	return 0;
}

nv_gpu_ringbuf_map_impl::~nv_gpu_ringbuf_map_impl()
{
	if (auto err = cuMemFree((CUdeviceptr)server_shared_mem);
	    err != CUDA_SUCCESS) {
		SPDLOG_WARN(
			"Unable to free memory for nv_gpu_ringbuf_map_impl");
	}
}
