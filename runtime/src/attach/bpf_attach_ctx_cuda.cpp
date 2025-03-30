#include "bpftime_shm_internal.hpp"
#include <bpf_attach_ctx.hpp>
#include "demo_ptx_prog.hpp"
#include <spdlog/spdlog.h>
#include "cuda.h"
#include "cupti_activity.h"
#define CUPTI_CALL(call, error_message)                                        \
	do {                                                                   \
		CUptiResult _status = call;                                    \
		if (_status != CUPTI_SUCCESS) {                                \
			const char *errstr;                                    \
			cuptiGetResultString(_status, &errstr);                \
			SPDLOG_ERROR("CUPTI Error: {}", errstr);               \
			throw std::runtime_error(error_message);               \
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

static void CUPTIAPI cupti_buffer_requested(uint8_t **buffer, size_t *size,
					    size_t *maxNumRecords)
{
	*size = 16 * 1024;
	*buffer = new uint8_t[*size];
	*maxNumRecords = 10;
}
static void CUPTIAPI cupti_buffer_completed(CUcontext ctx, uint32_t streamId,
					    uint8_t *buffer, size_t size,
					    size_t validSize)
{
	CUpti_Activity *record = nullptr;

	do {
		auto status =
			cuptiActivityGetNextRecord(buffer, validSize, &record);
		if (status == CUPTI_SUCCESS) {
			if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
				auto kernelActivity =
					(CUpti_ActivityKernel4 *)record;
				SPDLOG_INFO("Kernel {} consumed {} ns",
					    (const char *)kernelActivity->name,
					    kernelActivity->end -
						    kernelActivity->start);
			}
		} else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
			break;
		} else if (status == CUPTI_ERROR_INVALID_KIND) {
			break;
		} else {
			const char *error_str;
			cuptiGetResultString(status, &error_str);
			SPDLOG_ERROR("Unable to read more cupti records: {}",
				     error_str);
		}
	} while (true);
	delete[] buffer;
}

namespace bpftime
{

void bpf_attach_ctx::start_cuda_watcher_thread()
{
	auto flag = this->cuda_ctx->cuda_watcher_should_stop;
	std::thread([=, this]() {
		auto &ctx = cuda_ctx;
		while (!flag->load()) {
			const auto &array = ctx->cuda_shared_mem->time_sum;
			for (size_t i = 0; i < std::size(array); i++) {
				uint64_t cuda_time_sum = __atomic_load_n(
					&array[i], __ATOMIC_SEQ_CST);
				auto host_time_sum =
					ctx->operation_time_sum->at(i).load();

				SPDLOG_INFO(
					"Operation {} cuda_time_sum = {}, host_time_sum = {}, diff = {}",
					i, cuda_time_sum, host_time_sum,
					cuda_time_sum - host_time_sum);
			}
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
	}).detach();
	std::thread handle([=, this]() {
		SPDLOG_INFO("CUDA watcher thread started");
		auto &ctx = cuda_ctx;

		while (!flag->load()) {
			if (ctx->cuda_shared_mem->flag1 == 1) {
				ctx->cuda_shared_mem->flag1 = 0;
				auto req_id = ctx->cuda_shared_mem->request_id;

				auto map_ptr = ctx->cuda_shared_mem->map_id;
				auto map_fd = map_ptr >> 32;
				SPDLOG_DEBUG(
					"CUDA Received call request id {}, map_ptr = {}, map_fd = {}",
					req_id, map_ptr, map_fd);
				auto start_time =
					std::chrono::high_resolution_clock::now();
				if (req_id == (int)cuda::MapOperation::LOOKUP) {
					const auto &req =
						ctx->cuda_shared_mem->req
							.map_lookup;
					auto &resp = ctx->cuda_shared_mem->resp
							     .map_lookup;
					auto ptr = bpftime_map_lookup_elem(
						map_fd, req.key);
					resp.value = ptr;
					SPDLOG_DEBUG(
						"CUDA: Executing map lookup for {}, result = {}",
						map_fd, (uintptr_t)resp.value);
				} else if (req_id ==
					   (int)cuda::MapOperation::UPDATE) {
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
					   (int)cuda::MapOperation::DELETE) {
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
				} else if (req_id == 1000) {
					SPDLOG_INFO("Request probing..");

				} else {
					SPDLOG_WARN("Unknown request id {}",
						    req_id);
				}

				auto end_time =
					std::chrono::high_resolution_clock::now();
				if ((size_t)req_id <
				    ctx->operation_time_sum->size()) {
					std::chrono::duration<uint64_t,
							      std::nano>
						elasped_nanosecond =
							end_time - start_time;
					ctx->operation_time_sum->at(req_id)
						.fetch_add(elasped_nanosecond
								   .count());
				}
				ctx->cuda_shared_mem->flag2 = 1;
				std::atomic_thread_fence(
					std::memory_order_seq_cst);
			}
			std::this_thread::sleep_for(
				std::chrono::milliseconds(10));
		}
		SPDLOG_INFO("Exiting CUDA watcher thread");
	});
	handle.detach();
}

int bpf_attach_ctx::start_cuda_prober(int id)
{
	SPDLOG_DEBUG("Try starting CUDA program at {}", id);
	auto itr = instantiated_progs.find(id);
	if (itr == instantiated_progs.end()) {
		SPDLOG_ERROR("Invalid cuda program id: {}", id);
		return -1;
	}
	auto prog = *itr->second;
	if (!prog.is_cuda()) {
		SPDLOG_ERROR("Program id {} is not a CUDA program", id);
		return -1;
	}
	// Initialize CUPTI
	{
		CUPTI_CALL(
			cuptiActivityRegisterCallbacks(cupti_buffer_requested,
						       cupti_buffer_completed),
			"register cupti callbacks");
		SPDLOG_INFO("CUPTI callbacks registered");
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL),
			   "enable cupti event");
		SPDLOG_INFO("CUPTI initialized");
	}
	CUmodule raw_module;
	NV_SAFE_CALL_2(cuModuleLoadDataEx(&raw_module,
					  prog.get_cuda_elf_binary(), 0, 0, 0),
		       "Load CUDA module");
	SPDLOG_INFO("CUDA module loaded");
	cuda_ctx->set_module(raw_module);
	// Setup shared data pointer
	{
		CUdeviceptr constDataPtr;
		size_t constDataLen;

		NV_SAFE_CALL_2(cuModuleGetGlobal(&constDataPtr, &constDataLen,
						 raw_module, "constData"),
			       "Unable to find constData section");
		SPDLOG_INFO(
			"CUDA binary constData device pointer: {}, constData size: {}",
			(uintptr_t)constDataPtr, constDataLen);
		uintptr_t shared_mem_dev_ptr =
			cuda_ctx->cuda_shared_mem_device_pointer;
		NV_SAFE_CALL_2(cuMemcpyHtoD(constDataPtr, &shared_mem_dev_ptr,
					    sizeof(shared_mem_dev_ptr)),
			       "Copy device pointer value to device");
		SPDLOG_INFO("CUDA: constData set done");
	}
	// Setup map_info data
	{
		CUdeviceptr map_info_ptr;
		size_t map_info_len;
		NV_SAFE_CALL_2(cuModuleGetGlobal(&map_info_ptr, &map_info_len,
						 raw_module, "map_info"),
			       "Unable to get map_info handle");
		std::vector<cuda::MapBasicInfo> local_basic_info(256);
		if (sizeof(cuda::MapBasicInfo) * local_basic_info.size() !=
		    map_info_len) {
			SPDLOG_ERROR(
				"Unexpected map_info_len: {}, should be {}*{}",
				map_info_len, sizeof(cuda::MapBasicInfo),
				local_basic_info.size());
			return -1;
		}
		for (auto &entry : local_basic_info) {
			entry.enabled = false;
			entry.key_size = 0;
			entry.value_size = 0;
			entry.max_entries = 0;
		}
		const auto &handler_manager =
			*shm_holder.global_shared_memory.get_manager();
		for (size_t i = 0; i < handler_manager.size(); i++) {
			const auto &current_handler =
				handler_manager.get_handler(i);
			if (std::holds_alternative<bpf_map_handler>(
				    current_handler)) {
				auto &local = local_basic_info[i];
				SPDLOG_INFO(
					"Copying map fd {} to device, key size={}, value size={}, max ent={}",
					i, local.key_size, local.value_size,
					local.max_entries);
				const auto &map = std::get<bpf_map_handler>(
					current_handler);
				if (i >= local_basic_info.size()) {
					SPDLOG_ERROR(
						"Too large map fd: {}, max to be {}",
						i, local_basic_info.size());
					return -1;
				}

				local.enabled = true;
				local.key_size = map.get_key_size();
				local.value_size = map.get_value_size();
				local.max_entries = map.get_max_entries();
			}
		}
		NV_SAFE_CALL_2(cuMemcpyHtoD(map_info_ptr,
					    local_basic_info.data(),
					    map_info_len),
			       "Copy map_info to device");
		SPDLOG_INFO("CUDA: map_info set done");
	}
	CUfunction kernel;
	NV_SAFE_CALL_2(cuModuleGetFunction(&kernel, raw_module, "bpf_main"),
		       "get CUDA kernel function");
	CUdeviceptr arg1 = 0;
	uint64_t arg2 = 0;
	void *args[2] = { &arg1, &arg2 };
	NV_SAFE_CALL_2(cuLaunchKernel(kernel, 1, 1, 1, // grid dim
				      1, 1, 1, // block dim
				      0, nullptr, // shared mem and stream
				      args, 0),
		       "Unable to start kernel"); // arguments
	SPDLOG_INFO("CUDA program started..");
	std::thread handle([=, this]() {
		NV_SAFE_CALL(
			cuCtxSetCurrent(this->cuda_ctx->ctx_container.get()),
			"Unable to set CUDA context");
		if (auto err = cuCtxSynchronize(); err != CUDA_SUCCESS) {
			SPDLOG_ERROR("Unable to synchronize CUDA context: {}",
				     (int)err);
		} else {
			SPDLOG_INFO("CUDA kernel exited..");
		}
		CUPTI_CALL(cuptiActivityFlushAll(0),
			   "flushing cupti activities");
	});
	handle.detach();
	return 0;
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

std::optional<std::unique_ptr<cuda::CUDAContext> > create_cuda_context()
{
	NV_SAFE_CALL(cuInit(0), "Unable to initialize CUDA");
	SPDLOG_INFO("Initializing CUDA shared memory");
	auto cuda_shared_mem = std::make_unique<cuda::SharedMem>();
	memset(cuda_shared_mem.get(), 0, sizeof(*cuda_shared_mem));
	NV_SAFE_CALL(cuMemHostRegister(cuda_shared_mem.get(),
				       sizeof(cuda::SharedMem),
				       CU_MEMHOSTREGISTER_DEVICEMAP),
		     "Unable to register shared memory");
	CUdeviceptr memDevPtr;
	NV_SAFE_CALL(cuMemHostGetDevicePointer(&memDevPtr,
					       cuda_shared_mem.get(), 0),
		     "Unable to get device pointer");
	SPDLOG_INFO("CUDA shared memory addr: {}",
		    (uintptr_t)cuda_shared_mem.get());
	CUdevice device;
	NV_SAFE_CALL(cuDeviceGet(&device, 0), "Unable to get CUDA device");

	CUcontext raw_ctx;
	NV_SAFE_CALL(cuCtxCreate(&raw_ctx, 0, device), "Create CUDA context");
	auto cuda_ctx = std::make_optional(std::make_unique<cuda::CUDAContext>(
		std::move(cuda_shared_mem), raw_ctx));
	SPDLOG_INFO("CUDA context created");
	return cuda_ctx;
}
CUDAContext::~CUDAContext()
{
	SPDLOG_INFO("Destructing CUDAContext");
	if (auto result = cuMemHostUnregister(cuda_shared_mem.get());
	    result != CUDA_SUCCESS) {
		SPDLOG_ERROR("Unable to unregister host memory: {}",
			     (int)result);
	}
}
CUDAContext::CUDAContext(std::unique_ptr<cuda::SharedMem> &&mem,
			 CUcontext raw_ctx)
	: cuda_shared_mem(std::move(mem)),
	  cuda_shared_mem_device_pointer((uintptr_t)cuda_shared_mem.get()),
	  ctx_container(raw_ctx, cuda_context_destroyer),
	  operation_time_sum(
		  std::make_unique<std::array<std::atomic<uint64_t>, 8> >())
{
}
} // namespace cuda
} // namespace bpftime
