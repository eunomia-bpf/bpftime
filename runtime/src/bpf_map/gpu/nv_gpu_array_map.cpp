#include "nv_gpu_array_map.hpp"
#include "bpftime_internal.h"
#include "cuda.h"
#include "linux/bpf.h"
#include "spdlog/spdlog.h"
#include <cerrno>
#include <cstdint>
#include <stdexcept>

using namespace bpftime;

nv_gpu_array_map_impl::nv_gpu_array_map_impl(
	boost::interprocess::managed_shared_memory &memory,
	CUipcMemHandle gpu_mem_handle, uint64_t value_size,
	uint64_t max_entries, uint64_t thread_count)

	: gpu_mem_handle(gpu_mem_handle), value_size(value_size),
	  max_entries(max_entries), thread_count(thread_count),
	  value_buffer(memory.get_segment_manager())
{
	entry_size = thread_count * value_size;
	value_buffer.resize(entry_size);
	if (auto err = cuIpcOpenMemHandle(&gpu_shared_mem, gpu_mem_handle,
					  CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
	    err != CUDA_SUCCESS) {
		SPDLOG_ERROR("Unable to map CUDA IPC memory, error={}",
			     (int)err);
		throw std::runtime_error("Unable to map CUDA IPC memory!");
	}
}

void *nv_gpu_array_map_impl::elem_lookup(const void *key)
{
	auto key_val = *(uint32_t *)key;
	if (key_val >= max_entries) {
		errno = ENOENT;
		return nullptr;
	}
	if (CUresult err = cuMemcpyDtoH(value_buffer.data(),
					(CUdeviceptr)gpu_shared_mem +
						key_val * entry_size,
					entry_size);
	    err != CUDA_SUCCESS) {
		SPDLOG_ERROR("Unable to copy bytes from GPU to host: {}",
			     (int)err);
		return nullptr;
	}
	return value_buffer.data();
}

long nv_gpu_array_map_impl::elem_update(const void *key, const void *value,
					uint64_t flags)
{
	if (unlikely(!check_update_flags(flags)))
		return -1;
	auto key_val = *(uint32_t *)key;
	if (unlikely(key_val < max_entries && flags == BPF_NOEXIST)) {
		errno = EEXIST;
		return -1;
	}

	if (unlikely(key_val >= max_entries)) {
		errno = E2BIG;
		return -1;
	}
	if (auto err = cuMemcpyHtoD((CUdeviceptr)gpu_shared_mem +
					    key_val * entry_size,
				    value, entry_size);
	    err != CUDA_SUCCESS) {
		SPDLOG_ERROR("Unable to copy {} bytes from host to GPU: {}",
			     entry_size, (int)err);
		errno = EINVAL;
		return -1;
	}
	return 0;
}

long nv_gpu_array_map_impl::elem_delete(const void *key)
{
	errno = EINVAL;
	return -1;
}

int nv_gpu_array_map_impl::map_get_next_key(const void *key, void *next_key)
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

nv_gpu_array_map_impl::~nv_gpu_array_map_impl()
{
	if (auto err = cuIpcCloseMemHandle(gpu_shared_mem);
	    err != CUDA_SUCCESS) {
		SPDLOG_WARN(
			"Unable to release CUDA IPC handle when destroying nv_gpu_arracy_map_impl: {}",
			(int)err);
	}
}
