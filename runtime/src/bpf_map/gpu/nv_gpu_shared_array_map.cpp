#include "nv_gpu_shared_array_map.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include "cuda.h"
#include "linux/bpf.h"
#include "nv_gpu_gdrcopy.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <unistd.h>
#include <vector>

using namespace bpftime;

namespace
{

struct gpu_shared_array_local_entry {
	CUcontext cuda_ctx;
	CUstream agent_stream;
	CUdeviceptr mapped_ptr;
	bool is_owner;
};

shm_local_table<gpu_shared_array_local_entry> g_gpu_shared_array_local_table{};
static_assert(std::is_trivially_destructible_v<
	      decltype(g_gpu_shared_array_local_table)>);

void cleanup_gpu_shared_array_agent_local(
	gpu_shared_array_local_entry &entry) noexcept
{
	if (entry.is_owner)
		return;
	CUcontext prev_ctx = nullptr;
	bool did_switch_ctx = false;
	if (entry.cuda_ctx != nullptr) {
		cuCtxGetCurrent(&prev_ctx);
		if (prev_ctx != entry.cuda_ctx) {
			cuCtxSetCurrent(entry.cuda_ctx);
			did_switch_ctx = true;
		}
	}
	if (entry.agent_stream != nullptr) {
		cuStreamDestroy(entry.agent_stream);
		entry.agent_stream = nullptr;
	}
	if (entry.mapped_ptr != 0) {
		cuIpcCloseMemHandle(entry.mapped_ptr);
		entry.mapped_ptr = 0;
	}
	if (did_switch_ctx)
		cuCtxSetCurrent(prev_ctx);
}

gpu_shared_array_local_entry *
resolve_gpu_shared_array_agent_local(shm_local_key key,
				     const CUipcMemHandle &gpu_mem_handle)
{
	return g_gpu_shared_array_local_table.get_or_init(
		key,
		[&](gpu_shared_array_local_entry &out) -> bool {
			out.is_owner = false;
			out.cuda_ctx = nullptr;
			out.agent_stream = nullptr;
			out.mapped_ptr = 0;

			if (auto err = cuInit(0); err != CUDA_SUCCESS) {
				SPDLOG_ERROR("cuInit failed: {}", (int)err);
				errno = EINVAL;
				return false;
			}

			std::vector<int> candidates;
			if (const char *p = getenv("BPFTIME_CUDA_DEVICE");
			    p && p[0] != '\0') {
				errno = 0;
				char *end = nullptr;
				long v = strtol(p, &end, 10);
				if (errno == 0 && end && end[0] == '\0' &&
				    v >= 0 &&
				    v <= std::numeric_limits<int>::max()) {
					candidates.push_back((int)v);
				}
			}
			int count = 0;
			if (cuDeviceGetCount(&count) == CUDA_SUCCESS &&
			    count > 0) {
				for (int i = 0; i < count; i++)
					candidates.push_back(i);
			} else {
				candidates.push_back(0);
			}
			std::sort(candidates.begin(), candidates.end());
			candidates.erase(
				std::unique(candidates.begin(),
					    candidates.end()),
				candidates.end());

			CUdeviceptr ptr = 0;
			CUcontext chosen_ctx = nullptr;
			CUcontext prev_ctx = nullptr;
			cuCtxGetCurrent(&prev_ctx);
			if (prev_ctx != nullptr) {
				auto err2 = cuIpcOpenMemHandle(
					&ptr, gpu_mem_handle,
					CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
				if (err2 == CUDA_SUCCESS) {
					chosen_ctx = prev_ctx;
				} else {
					ptr = 0;
				}
			}

			if (ptr == 0) {
				for (int d : candidates) {
					CUdevice dev = 0;
					if (cuDeviceGet(&dev, d) !=
					    CUDA_SUCCESS)
						continue;
					CUcontext ctx = nullptr;
					if (cuDevicePrimaryCtxRetain(&ctx,
								     dev) !=
						    CUDA_SUCCESS ||
					    ctx == nullptr)
						continue;
					if (cuCtxSetCurrent(ctx) !=
					    CUDA_SUCCESS) {
						cuDevicePrimaryCtxRelease(dev);
						continue;
					}
					auto err2 = cuIpcOpenMemHandle(
						&ptr, gpu_mem_handle,
						CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
					if (err2 == CUDA_SUCCESS && ptr != 0) {
						chosen_ctx = ctx;
						SPDLOG_INFO(
							"Mapped CUDA IPC handle on device {}",
							d);
						break;
					}
					ptr = 0;
					cuDevicePrimaryCtxRelease(dev);
				}
			}

			if (prev_ctx != nullptr)
				cuCtxSetCurrent(prev_ctx);

			if (ptr == 0 || chosen_ctx == nullptr) {
				SPDLOG_ERROR(
					"Unable to map CUDA IPC memory for shared array map on any device");
				errno = EINVAL;
				return false;
			}

			CUcontext tmp_prev = nullptr;
			cuCtxGetCurrent(&tmp_prev);
			if (tmp_prev != chosen_ctx)
				cuCtxSetCurrent(chosen_ctx);
			CUstream stream = nullptr;
			CUresult err2 = cuStreamCreate(&stream,
						       CU_STREAM_NON_BLOCKING);
			if (tmp_prev != chosen_ctx)
				cuCtxSetCurrent(tmp_prev);
			if (err2 != CUDA_SUCCESS || stream == nullptr) {
				SPDLOG_ERROR(
					"Unable to create CUDA stream for agent: {}",
					(int)err2);
				errno = EINVAL;
				return false;
			}

			out.cuda_ctx = chosen_ctx;
			out.agent_stream = stream;
			out.mapped_ptr = ptr;
			return true;
		},
		cleanup_gpu_shared_array_agent_local);
}

} // namespace

nv_gpu_shared_array_map_impl::nv_gpu_shared_array_map_impl(
	const shm_init_ctx &ctx, uint64_t value_size, uint64_t max_entries)
	: server_gpu_shared_mem(0), value_size(value_size),
	  max_entries(max_entries), value_buffer(ctx.segment_manager())
{
	CUcontext owner_ctx = nullptr;
	if (shm_holder.global_shared_memory.get_open_type() !=
	    shm_open_type::SHM_OPEN_ONLY) {
		cuCtxGetCurrent(&owner_ctx);
		if (!owner_ctx) {
			SPDLOG_WARN(
				"Owner CUDA context not set at init; CUDA calls may fail");
		}
	}
	value_buffer.resize(value_size);
	local_key_ = ctx.allocate_local_key();
	auto total_buffer_size = (uint64_t)value_size * max_entries;
	SPDLOG_INFO(
		"Initializing map type of BPF_MAP_TYPE_GPU_ARRAY_MAP (device), total_buffer_size={}",
		total_buffer_size);
	if (auto err = cuMemAlloc(&server_gpu_shared_mem, total_buffer_size);
	    err != CUDA_SUCCESS) {
		release_local_key(
			value_buffer.get_allocator().get_segment_manager(),
			local_key_);
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
		cuMemFree(server_gpu_shared_mem);
		server_gpu_shared_mem = 0;
		release_local_key(
			value_buffer.get_allocator().get_segment_manager(),
			local_key_);
		SPDLOG_ERROR(
			"Unable to open CUDA IPC handle for nv_gpu_shared_array_map_impl: {}",
			(int)err);
		throw std::runtime_error(
			"Unable to open CUDA IPC handle for nv_gpu_shared_array_map_impl");
	}
	if (shm_holder.global_shared_memory.get_open_type() !=
	    shm_open_type::SHM_OPEN_ONLY) {
		auto *owner_entry =
			g_gpu_shared_array_local_table.get_or_init(
				local_key_,
				[&](gpu_shared_array_local_entry &out) -> bool {
					out.cuda_ctx = owner_ctx;
					out.agent_stream = nullptr;
					out.mapped_ptr = server_gpu_shared_mem;
					out.is_owner = true;
					return true;
				});
		if (owner_entry == nullptr) {
			if (cuMemFree(server_gpu_shared_mem) != CUDA_SUCCESS)
				SPDLOG_WARN(
					"Unable to free CUDA memory after local table exhaustion");
			server_gpu_shared_mem = 0;
			release_local_key(
				value_buffer.get_allocator()
					.get_segment_manager(),
				local_key_);
			throw std::runtime_error(
				"GPU shared array local table slot unavailable");
		}
	}
}

void *nv_gpu_shared_array_map_impl::elem_lookup(const void *key)
{
	auto key_val = *(const uint32_t *)key;
	if (key_val >= max_entries) {
		errno = ENOENT;
		return nullptr;
	}
	const bool is_agent = shm_holder.global_shared_memory.get_open_type() ==
			      shm_open_type::SHM_OPEN_ONLY;
	gpu_shared_array_local_entry *local =
		is_agent ? resolve_gpu_shared_array_agent_local(local_key_,
								gpu_mem_handle) :
			   g_gpu_shared_array_local_table.find(local_key_);
	CUdeviceptr base =
		is_agent ? (local != nullptr ? local->mapped_ptr : 0) :
			   server_gpu_shared_mem;
	if (is_agent &&
	    (base == 0 || local == nullptr || local->cuda_ctx == nullptr)) {
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
	CUcontext target_ctx = local != nullptr ? local->cuda_ctx : nullptr;
	cuCtxPushCurrent_v2(target_ctx);

	CUresult err = CUDA_SUCCESS;
	if (is_agent) {
		cuMemcpyDtoHAsync_v2(
			value_buffer.data(),
			(CUdeviceptr)base + (uint64_t)key_val * value_size,
			value_size, local->agent_stream);
		err = cuStreamSynchronize(local->agent_stream);
	} else {
		err = cuMemcpyDtoH(value_buffer.data(),
				   (CUdeviceptr)base +
					   (uint64_t)key_val * value_size,
				   value_size);
	}

	if (err != CUDA_SUCCESS) {
		SPDLOG_ERROR(
			"Unable to copy bytes from GPU to host: {}", (int)err);
		cuCtxPopCurrent(&target_ctx);
		return nullptr;
	}
	cuCtxPopCurrent(&target_ctx);
	return value_buffer.data();
}

long nv_gpu_shared_array_map_impl::elem_update(const void *key,
					       const void *value,
					       uint64_t flags)
{
	if (!bpftime::check_update_flags(flags))
		return -1;
	auto key_val = *(const uint32_t *)key;
	if ((key_val < max_entries) && flags == BPF_NOEXIST) {
		errno = EEXIST;
		return -1;
	}
	if (key_val >= max_entries) {
		errno = E2BIG;
		return -1;
	}
	const bool is_agent = shm_holder.global_shared_memory.get_open_type() ==
			      shm_open_type::SHM_OPEN_ONLY;
	gpu_shared_array_local_entry *local =
		is_agent ? resolve_gpu_shared_array_agent_local(local_key_,
								gpu_mem_handle) :
			   g_gpu_shared_array_local_table.find(local_key_);
	CUdeviceptr base =
		is_agent ? (local != nullptr ? local->mapped_ptr : 0) :
			   server_gpu_shared_mem;
	if (is_agent &&
	    (base == 0 || local == nullptr || local->cuda_ctx == nullptr)) {
		errno = EINVAL;
		return -1;
	}
	CUcontext target_ctx = local != nullptr ? local->cuda_ctx : nullptr;
	cuCtxPushCurrent_v2(target_ctx);

	CUresult err = CUDA_SUCCESS;
	if (is_agent) {
		cuMemcpyHtoDAsync_v2((CUdeviceptr)base +
					     (uint64_t)key_val * value_size,
				     value, value_size, local->agent_stream);
		err = cuStreamSynchronize(local->agent_stream);
	} else {
		err = cuMemcpyHtoD((CUdeviceptr)base +
					   (uint64_t)key_val * value_size,
				   value, value_size);
	}

	if (err != CUDA_SUCCESS) {
		SPDLOG_ERROR(
			"Unable to copy {} bytes from host to GPU: {}",
			value_size, (int)err);
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
	}
	auto key_val = *(const uint32_t *)key;
	if (key_val >= max_entries) {
		next_key_val = 0;
		return 0;
	}
	if (key_val + 1 == max_entries) {
		errno = ENOENT;
		return -1;
	}
	next_key_val = key_val + 1;
	return 0;
}

nv_gpu_shared_array_map_impl::~nv_gpu_shared_array_map_impl()
{
	auto total_buffer_size = (uint64_t)value_size * max_entries;
	bpftime::gpu::gdrcopy::destroy_gdrcopy_mapping_for_owner(
		this, "BPF_MAP_TYPE_GPU_ARRAY_MAP", total_buffer_size);
	g_gpu_shared_array_local_table.evict(
		local_key_, [&](gpu_shared_array_local_entry &entry) {
			if (entry.is_owner) {
				CUcontext prev_ctx = nullptr;
				bool did_switch_ctx = false;
				if (entry.cuda_ctx) {
					cuCtxGetCurrent(&prev_ctx);
					if (prev_ctx != entry.cuda_ctx) {
						cuCtxSetCurrent(entry.cuda_ctx);
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
				cleanup_gpu_shared_array_agent_local(entry);
			}
		});
	release_local_key(value_buffer.get_allocator().get_segment_manager(),
			  local_key_);
}

CUdeviceptr
nv_gpu_shared_array_map_impl::try_initialize_for_agent_and_get_mapped_address()
{
	if (shm_holder.global_shared_memory.get_open_type() ==
	    shm_open_type::SHM_OPEN_ONLY) {
		auto *local = resolve_gpu_shared_array_agent_local(
			local_key_, gpu_mem_handle);
		return local != nullptr ? local->mapped_ptr : 0;
	}
	return server_gpu_shared_mem;
}
