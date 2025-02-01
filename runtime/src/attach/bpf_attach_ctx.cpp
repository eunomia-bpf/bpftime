/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "attach_private_data.hpp"
#include "base_attach_impl.hpp"
#include "bpftime_shm.hpp"
#include "cuda.h"
#include "handler/link_handler.hpp"
#include "handler/map_handler.hpp"
#include "handler/prog_handler.hpp"
#include "nvPTXCompiler.h"
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <map>
#include <memory>
#include <syscall_table.hpp>
#include <bpf_attach_ctx.hpp>
#include <bpftime_shm_internal.hpp>
#include <bpftime_prog.hpp>
#include "bpftime_config.hpp"
#include <spdlog/spdlog.h>
#include <handler/perf_event_handler.hpp>
#include <bpftime_helper_group.hpp>
#include <handler/handler_manager.hpp>
#include <tuple>
#include <utility>
#include <variant>
#include <sys/resource.h>

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

extern "C" uint64_t bpftime_set_retval(uint64_t value);
namespace bpftime
{

static int load_prog_and_helpers(bpftime_prog *prog, const agent_config &config)
{
#if __linux__
	if (config.enable_kernel_helper_group) {
		bpftime_helper_group::get_kernel_utils_helper_group()
			.add_helper_group_to_prog(prog);
	}
#endif
	if (config.enable_ufunc_helper_group) {
		bpftime_helper_group::get_ufunc_helper_group()
			.add_helper_group_to_prog(prog);
	}
	if (config.enable_shm_maps_helper_group) {
		bpftime_helper_group::get_shm_maps_helper_group()
			.add_helper_group_to_prog(prog);
	}
	return prog->bpftime_prog_load(config.jit_enabled);
}

int bpf_attach_ctx::init_attach_ctx_from_handlers(const agent_config &config)
{
	const handler_manager *manager =
		shm_holder.global_shared_memory.get_manager();
	if (!manager) {
		return -1;
	}
	return init_attach_ctx_from_handlers(manager, config);
}

// create a attach context and progs from handlers
int bpf_attach_ctx::init_attach_ctx_from_handlers(
	const handler_manager *manager, const agent_config &config)
{
	for (int i = 0; i < (int)manager->size(); i++) {
		if (manager->is_allocated(i)) {
			std::set<int> stk;
			if (int err = instantiate_handler_at(manager, i, stk,
							     config);
			    err < 0) {
				SPDLOG_INFO("Failed to instantiate handler {}",
					    i);
				// Unable to instantiate handler may not be an
				// error. We can continue trying to instantiate
				// other handlers.
			}
		}
	}
	return 0;
}

bpf_attach_ctx::~bpf_attach_ctx()
{
	SPDLOG_DEBUG("Destructor: bpf_attach_ctx");

	cuda_ctx->cuda_watcher_should_stop->store(true);
}

// create a probe context
bpf_attach_ctx::bpf_attach_ctx() : cuda_ctx(*cuda::create_cuda_context())
{
	current_id = CURRENT_ID_OFFSET;
	SPDLOG_INFO("bpf_attach_ctx constructed");
	start_cuda_watcher_thread();
}

int bpf_attach_ctx::instantiate_handler_at(const handler_manager *manager,
					   int id, std::set<int> &stk,
					   const agent_config &config)
{
	SPDLOG_DEBUG("Instantiating handler at {}", id);
	if (instantiated_handlers.contains(id)) {
		SPDLOG_DEBUG("Handler {} already instantiated", id);
		return 0;
	}
	if (stk.contains(id)) {
		SPDLOG_CRITICAL("Loop detected when instantiating handler {}",
				id);
		return -1;
	}
	stk.insert(id);
	auto &handler = manager->get_handler(id);
	if (std::holds_alternative<bpf_prog_handler>(handler)) {
		if (int err = instantiate_prog_handler_at(
			    id, std::get<bpf_prog_handler>(handler), config);
		    err < 0) {
			SPDLOG_ERROR(
				"Unable to instantiate bpf prog handler {}: {}",
				id, err);
			return err;
		}
	} else if (std::holds_alternative<bpf_perf_event_handler>(handler)) {
		if (int err = instantiate_perf_event_handler_at(
			    id, std::get<bpf_perf_event_handler>(handler));
		    err < 0) {
			SPDLOG_ERROR(
				"Unable to instantiate bpf perf event handler {}: {}",
				id, err);
			return err;
		}
	} else if (std::holds_alternative<bpf_link_handler>(handler)) {
		auto &link_handler = std::get<bpf_link_handler>(handler);
		if (int err = instantiate_handler_at(
			    manager, link_handler.prog_id, stk, config);
		    err < 0) {
			SPDLOG_ERROR(
				"Unable to instantiate prog handler {} when instantiating link handler {}: {}",
				link_handler.prog_id, id, err);
			return err;
		}
		if (int err = instantiate_handler_at(
			    manager, link_handler.attach_target_id, stk,
			    config);
		    err < 0) {
			SPDLOG_ERROR(
				"Unable to instantiate perf event handler {} when instantiating link handler {}: {}",
				link_handler.attach_target_id, id, err);
			return err;
		}
		if (int err = instantiate_bpf_link_handler_at(id, link_handler);
		    err < 0) {
			SPDLOG_DEBUG(
				"Unable to instantiate bpf link handler {}: {}",
				id, err);
			return err;
		}
	} else {
		SPDLOG_DEBUG("Instantiating type {}", handler.index());
	}
	stk.erase(id);
	instantiated_handlers.insert(id);
	SPDLOG_DEBUG("Instantiating done: {}", id);
	return 0;
}

void bpf_attach_ctx::register_attach_impl(
	std::initializer_list<int> &&attach_types,
	std::unique_ptr<attach::base_attach_impl> &&impl,
	std::function<std::unique_ptr<attach::attach_private_data>(
		const std::string_view &, int &)>
		private_data_creator)
{
	impl->register_custom_helpers([&](unsigned int idx, const char *name,
					  void *func) -> int {
		SPDLOG_INFO("Register attach-impl defined helper {}, index {}",
			    name, idx);
		this->helpers[idx] = bpftime_helper_info{ .index = idx,
							  .name = name,
							  .fn = func };
		return 0;
	});
	auto *impl_ptr = impl.get();
	attach_impl_holders.emplace_back(std::move(impl));
	for (auto ty : attach_types) {
		SPDLOG_DEBUG("Register attach type {} with attach impl {}", ty,
			     typeid(impl_ptr).name());
		attach_impls[ty] =
			std::make_pair(impl_ptr, private_data_creator);
	}
}
int bpf_attach_ctx::instantiate_prog_handler_at(int id,
						const bpf_prog_handler &handler,
						const agent_config &config)
{
	const ebpf_inst *insns = handler.insns.data();
	size_t cnt = handler.insns.size();
	const char *name = handler.name.c_str();
	instantiated_progs[id] =
		std::make_unique<bpftime_prog>(insns, cnt, name);
	bpftime_prog *prog = instantiated_progs[id].get();
	if (int err = load_prog_and_helpers(prog, config); err < 0) {
		SPDLOG_ERROR(
			"Failed to load program helpers for prog handler {}: {}",
			id, err);
		return err;
	}
	for (const auto &item : helpers) {
		prog->bpftime_prog_register_raw_helper(item.second);
	}
	return 0;
}
int bpf_attach_ctx::instantiate_bpf_link_handler_at(
	int id, const bpf_link_handler &handler)
{
	SPDLOG_DEBUG(
		"Instantiating link handler: prog {} -> perf event {}, cookie {}",
		handler.prog_id, handler.attach_target_id,
		handler.attach_cookie.value_or(0));
	auto &[priv_data, attach_type] =
		instantiated_perf_events[handler.attach_target_id];
	attach::base_attach_impl *attach_impl;
	// Find what kind of attach type it is
	if (auto itr = attach_impls.find(attach_type);
	    itr != attach_impls.end()) {
		attach_impl = itr->second.first;
	} else {
		SPDLOG_ERROR("Attach type {} is not registered", attach_type);
		return -ENOTSUP;
	}
	auto prog = instantiated_progs.at(handler.prog_id).get();
	if (prog->is_cuda()) {
		SPDLOG_INFO("Handling link to CUDA program: {}", id);
		if (int err = start_cuda_program(handler.prog_id); err < 0) {
			SPDLOG_ERROR(
				"Unable to start CUDA program for link id {}, prog id {}",
				id, handler.prog_id);
			return err;
		}
		instantiated_attach_links[id] = std::make_pair(0, nullptr);

		return 0;
	}
	auto cookie = handler.attach_cookie;
	int attach_id = attach_impl->create_attach_with_ebpf_callback(
		[=](void *mem, size_t mem_size, uint64_t *ret) -> int {
			current_thread_bpf_cookie = cookie;
			int err = prog->bpftime_prog_exec((void *)mem, mem_size,
							  ret);
			return err;
		},
		*priv_data, attach_type);
	if (attach_id < 0) {
		// Since the agent might be attach to a unrelated process
		// Using LD_PRELOAD, it's not an error here.
		SPDLOG_DEBUG("Unable to instantiate bpf link handler {}: {}",
			     id, attach_id);
		return attach_id;
	}
	instantiated_attach_links[id] = std::make_pair(attach_id, attach_impl);
	return 0;
}
int bpf_attach_ctx::instantiate_perf_event_handler_at(
	int id, const bpf_perf_event_handler &perf_handler)
{
	SPDLOG_DEBUG("Instantiating perf event handler at {}, type {}", id,
		     (int)perf_handler.type);
	if (perf_handler.type == (int)bpf_event_type::PERF_TYPE_SOFTWARE) {
		SPDLOG_DEBUG(
			"Detected software perf event at {}, nothing need to do",
			id);
		return 0;
	}
	std::unique_ptr<attach::attach_private_data> priv_data;

	auto itr = attach_impls.find((int)perf_handler.type);
	if (itr == attach_impls.end()) {
		SPDLOG_ERROR(
			"Unable to lookup attach implementation of attach type {}",
			(int)perf_handler.type);
		return -ENOENT;
	}
	auto &[attach_impl, private_data_gen] = itr->second;
	if (perf_handler.type ==
		    (int)bpf_event_type::BPF_TYPE_UPROBE_OVERRIDE ||
	    perf_handler.type == (int)bpf_event_type::BPF_TYPE_UPROBE ||
	    perf_handler.type == (int)bpf_event_type::BPF_TYPE_URETPROBE ||
	    perf_handler.type == (int)bpf_event_type::BPF_TYPE_UREPLACE) {
		auto &uprobe_data =
			std::get<uprobe_perf_event_data>(perf_handler.data);
		std::string arg_str;
		arg_str += uprobe_data._module_name;
		arg_str += ':';
		arg_str += std::to_string(uprobe_data.offset);
		int err = 0;
		priv_data = private_data_gen(arg_str, err);
		if (err < 0) {
			SPDLOG_ERROR(
				"Unable to parse private data of uprobe perf handler {}, arg_str `{}`: {}",
				id, arg_str, err);
			return err;
		}
	} else if (perf_handler.type ==
		   (int)bpf_event_type::PERF_TYPE_TRACEPOINT) {
		auto &tracepoint_data =
			std::get<tracepoint_perf_event_data>(perf_handler.data);
		int err = 0;
		priv_data = private_data_gen(
			std::to_string(tracepoint_data.tracepoint_id), err);
		if (err < 0) {
			SPDLOG_ERROR(
				"Unable to parse private data of tracepoint perf handler {}, tp_id `{}`: {}",
				id, tracepoint_data.tracepoint_id, err);
			return err;
		}
	} else {
		auto &custom_data =
			std::get<custom_perf_event_data>(perf_handler.data);
		int err = 0;
		priv_data = private_data_gen(
			std::string(custom_data.attach_argument), err);
		if (err < 0) {
			SPDLOG_ERROR(
				"Unable to parse private data of attach type {}, err={}, raw string={}",
				perf_handler.type, err,
				custom_data.attach_argument);
			return err;
		}
	}
	SPDLOG_DEBUG("Instantiated perf event handler {}", id);
	instantiated_perf_events[id] =
		std::make_pair(std::move(priv_data), (int)perf_handler.type);

	return 0;
}
int bpf_attach_ctx::destroy_instantiated_attach_link(int link_id)
{
	SPDLOG_DEBUG("Destroy attach link {}", link_id);
	if (auto itr = instantiated_attach_links.find(link_id);
	    itr != instantiated_attach_links.end()) {
		auto [attach_id, impl] = itr->second;
		if (impl == nullptr) {
			SPDLOG_INFO("Detach: Ignore attach with empty impl: {}",
				    link_id);
			return 0;
		}
		if (int err = impl->detach_by_id(attach_id); err < 0) {
			SPDLOG_ERROR(
				"Failed to detach attach link id {}, attach-specified id {}: {}",
				link_id, attach_id, err);
			return err;
		}
		instantiated_attach_links.erase(itr);
		return 0;
	} else {
		SPDLOG_ERROR("Unable to find instantiated attach link id {}",
			     link_id);
		return -ENOENT;
	}
}
int bpf_attach_ctx::destroy_all_attach_links()
{
	// Avoid modifying along with iterating..
	std::vector<int> to_detach;
	for (const auto &[k, _] : instantiated_attach_links)
		to_detach.push_back(k);
	for (auto k : to_detach) {
		SPDLOG_DEBUG("Destrying attach link {}", k);
		if (int err = destroy_instantiated_attach_link(k); err < 0) {
			SPDLOG_ERROR("Unable to destroy attach link {}: {}", k,
				     err);
			return err;
		}
	}
	return 0;
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

				auto map_ptr = ctx->cuda_shared_mem->map_id;
				auto map_fd = map_ptr >> 32;
				SPDLOG_DEBUG(
					"CUDA Received call request id {}, map_ptr = {}, map_fd = {}",
					req_id, map_ptr, map_fd);
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
				} else {
					SPDLOG_WARN("Unknown request id {}",
						    req_id);
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

int bpf_attach_ctx::start_cuda_program(int id)
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

	CUmodule raw_module;
	NV_SAFE_CALL_2(cuModuleLoadDataEx(&raw_module,
					  prog.get_cuda_elf_binary(), 0, 0, 0),
		       "Load CUDA module");
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
} // namespace cuda
} // namespace bpftime
