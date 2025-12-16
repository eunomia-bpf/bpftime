// #include "pos/cuda_impl/utils/fatbin.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "spdlog/spdlog.h"
#include "vector_types.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <frida-gum.h>
#include <iterator>
#include <memory>
#include <optional>
#include <vector>
#include "nv_attach_impl.hpp"
#include <stdexcept>
using namespace bpftime;
using namespace attach;

#define CUDA_DRIVER_CHECK_EXCEPTION(expr, message)                             \
	do {                                                                   \
		if (auto err = expr; err != CUDA_SUCCESS) {                    \
			SPDLOG_ERROR("{}: {}", message, (int)err);             \
			throw std::runtime_error(message);                     \
		}                                                              \
	} while (false)

extern "C" {

typedef struct __attribute__((__packed__)) fat_elf_header {
	uint32_t magic;
	uint16_t version;
	uint16_t header_size;
	uint64_t size;
} fat_elf_header_t;
}

typedef struct _CUDARuntimeFunctionHooker {
	GObject parent;
} CUDARuntimeFunctionHooker;

static void cuda_runtime_function_hooker_iface_init(gpointer g_iface,
						    gpointer iface_data);

// #define EXAMPLE_TYPE_LISTENER (cuda_runtime_function_hooker_iface_init())
G_DECLARE_FINAL_TYPE(CUDARuntimeFunctionHooker, cuda_runtime_function_hooker,
		     BPFTIME, NV_ATTACH_IMPL, GObject)
G_DEFINE_TYPE_EXTENDED(
	CUDARuntimeFunctionHooker, cuda_runtime_function_hooker, G_TYPE_OBJECT,
	0,
	G_IMPLEMENT_INTERFACE(GUM_TYPE_INVOCATION_LISTENER,
			      cuda_runtime_function_hooker_iface_init))

using cu_graph_add_kernel_node_v1_fn_t =
	CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t,
		     const CUDA_KERNEL_NODE_PARAMS_v1 *);
using cu_graph_add_kernel_node_v2_fn_t = decltype(&cuGraphAddKernelNode_v2);
using cu_graph_exec_kernel_node_set_params_v1_fn_t = CUresult (*)(
	CUgraphExec, CUgraphNode, const CUDA_KERNEL_NODE_PARAMS_v1 *);
using cu_graph_exec_kernel_node_set_params_v2_fn_t =
	decltype(&cuGraphExecKernelNodeSetParams_v2);
using cu_graph_kernel_node_set_params_v1_fn_t =
	CUresult (*)(CUgraphNode, const CUDA_KERNEL_NODE_PARAMS_v1 *);
using cu_graph_kernel_node_set_params_v2_fn_t =
	decltype(&cuGraphKernelNodeSetParams_v2);

using cuda_launch_kernel_fn_t = cudaError_t (*)(const void *, dim3, dim3,
						void **, size_t, cudaStream_t);

static bool cuda_graph_stream_is_capturing(cudaStream_t stream)
{
	cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
	auto err = cudaStreamIsCapturing(stream, &status);
	if (err != cudaSuccess)
		return true;
	return status != cudaStreamCaptureStatusNone;
}

static cudaError_t
cuda_launch_kernel_common(nv_attach_impl *impl, void *original_fn_ptr,
			  const void *func, dim3 grid_dim, dim3 block_dim,
			  void **args, size_t shared_mem, cudaStream_t stream)
{
	if (impl == nullptr)
		return cudaErrorUnknown;
	auto original =
		reinterpret_cast<cuda_launch_kernel_fn_t>(original_fn_ptr);
	if (!original) {
		SPDLOG_ERROR("Original cudaLaunchKernel function is null");
		return cudaErrorUnknown;
	}
	if (cuda_graph_stream_is_capturing(stream))
		return original(func, grid_dim, block_dim, args, shared_mem,
				stream);
	if (auto itr1 = impl->symbol_address_to_fatbin.find((void *)func);
	    itr1 != impl->symbol_address_to_fatbin.end()) {
		const auto &fatbin = *itr1->second;
		const auto &handle =
			fatbin.function_addr_to_symbol.at((void *)func);
		if (auto err = cuLaunchKernel(
			    handle.func, grid_dim.x, grid_dim.y, grid_dim.z,
			    block_dim.x, block_dim.y, block_dim.z, shared_mem,
			    stream, args, nullptr);
		    err != CUDA_SUCCESS) {
			const char *error_name = nullptr;
			const char *error_string = nullptr;
			cuGetErrorName(err, &error_name);
			cuGetErrorString(err, &error_string);
			SPDLOG_ERROR("Unable to launch kernel: {} ({})",
				     error_name ? error_name : "UNKNOWN",
				     error_string ? error_string :
						    "No description");
			SPDLOG_ERROR("Error code: {}", (int)err);
			return cudaErrorLaunchFailure;
		}
		return cudaSuccess;
	}
	return original(func, grid_dim, block_dim, args, shared_mem, stream);
}

static void example_listener_on_enter(GumInvocationListener *listener,
				      GumInvocationContext *ic)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto context =
		GUM_IC_GET_FUNC_DATA(ic, CUDARuntimeFunctionHookerContext *);
	if (context->to_function == AttachedToFunction::RegisterFatbin) {
		SPDLOG_DEBUG("Entering __cudaRegisterFatBinary..");

		auto header = (__fatBinC_Wrapper_t *)
			gum_invocation_context_get_nth_argument(gum_ctx, 0);
		auto data = (const char *)header->data;
		fat_elf_header_t *curr_header = (fat_elf_header_t *)data;
		const char *tail = (const char *)curr_header;
		while (true) {
			// #define FATBIN_TEXT_MAGIC 0xBA55ED50
			if (curr_header->magic == 0xBA55ED50) {
				SPDLOG_DEBUG(
					"Got CUBIN section header size = {}, size = {}",
					static_cast<int>(
						curr_header->header_size),
					static_cast<int>(curr_header->size));
				tail = ((const char *)curr_header) +
				       curr_header->header_size +
				       curr_header->size;
				curr_header = (fat_elf_header_t *)tail;
			} else {
				break;
			}
		};
		std::vector<uint8_t> data_vec((uint8_t *)data, (uint8_t *)tail);
		SPDLOG_INFO("Finally size = {}", data_vec.size());
		auto extracted_ptx =
			context->impl->extract_ptxs(std::move(data_vec));
		SPDLOG_INFO("Patching PTXs");
		auto fatbin_record = std::make_unique<struct fatbin_record>();
		fatbin_record->original_ptx = extracted_ptx;
		fatbin_record->module_pool = context->impl->module_pool;
		fatbin_record->ptx_pool = context->impl->ptx_pool;

		context->impl->current_fatbin = fatbin_record.get();
		context->impl->fatbin_records.emplace_back(
			std::move(fatbin_record));

	} else if (context->to_function ==
		   AttachedToFunction::RegisterFunction) {
		SPDLOG_DEBUG("Entering __cudaRegisterFunction..");
		auto &impl = *context->impl;
		auto current_fatbin = context->impl->current_fatbin;
		current_fatbin->try_loading_ptxs(*context->impl);

		auto func_addr =
			gum_invocation_context_get_nth_argument(gum_ctx, 1);
		auto symbol_name =
			(const char *)gum_invocation_context_get_nth_argument(
				gum_ctx, 3);
		if (auto ok = current_fatbin->find_and_fill_function_info(
			    func_addr, symbol_name);
		    !ok) {
			SPDLOG_WARN(
				"Unable to find_and_fill function info of symbol named {}, the PTX may not be compiled due to not modifying by nv_attach_impl",
				symbol_name);
		} else {
			context->impl->symbol_address_to_fatbin[func_addr] =
				current_fatbin;
				if (auto itr = current_fatbin->function_addr_to_symbol
						       .find(func_addr);
				    itr !=
				    current_fatbin->function_addr_to_symbol.end())
					impl.record_patched_kernel_function(
						std::string(symbol_name),
						itr->second.func);
			SPDLOG_DEBUG(
				"Registered kernel function name {} addr {:x}",
				symbol_name, (uintptr_t)func_addr);
		}

	} else if (context->to_function ==
		   AttachedToFunction::RegisterVariable) {
		SPDLOG_DEBUG("Entering __cudaRegisterVar");
		auto current_fatbin = context->impl->current_fatbin;
		current_fatbin->try_loading_ptxs(*context->impl);
		auto fatbin_handle =
			gum_invocation_context_get_nth_argument(gum_ctx, 0);
		auto var_addr =
			gum_invocation_context_get_nth_argument(gum_ctx, 1);
		auto symbol_name =
			(const char *)gum_invocation_context_get_nth_argument(
				gum_ctx, 3);
		SPDLOG_DEBUG("Registering variable named {}", symbol_name);

		if (bool ok = current_fatbin->find_and_fill_variable_info(
			    var_addr, symbol_name);
		    !ok) {
			SPDLOG_WARN(
				"Unable to find_and_fill variable info of symbol names {}, the PTX may not be compiled due to not modifying by nv_attach_impl",
				symbol_name);
		} else {
			context->impl->symbol_address_to_fatbin[var_addr] =
				current_fatbin;
			SPDLOG_DEBUG("Registered variable name {} addr {:x}",
				     symbol_name, (uintptr_t)var_addr);
		}

	} else if (context->to_function ==
		   AttachedToFunction::RegisterFatbinEnd) {
		SPDLOG_DEBUG("Entering __cudaRegisterFatBinaryEnd..");
		auto &current_fatbin = context->impl->current_fatbin;

		current_fatbin = nullptr;
	} else if (context->to_function == AttachedToFunction::CudaMalloc) {
		SPDLOG_DEBUG("Entering cudaMalloc..");
	} else if (context->to_function ==
			   AttachedToFunction::CudaMemcpyToSymbol ||
		   context->to_function ==
			   AttachedToFunction::CudaMemcpyToSymbolAsync) {
		auto symbol =
			(const void *)gum_invocation_context_get_nth_argument(
				gum_ctx, 0);
		auto src =
			(const void *)gum_invocation_context_get_nth_argument(
				gum_ctx, 1);
		auto count = static_cast<size_t>(reinterpret_cast<uintptr_t>(
			gum_invocation_context_get_nth_argument(gum_ctx, 2)));
		auto offset = static_cast<size_t>(reinterpret_cast<uintptr_t>(
			gum_invocation_context_get_nth_argument(gum_ctx, 3)));
		auto kind =
			static_cast<cudaMemcpyKind>(reinterpret_cast<uintptr_t>(
				gum_invocation_context_get_nth_argument(gum_ctx,
									4)));
		cudaStream_t stream = nullptr;
		bool async = context->to_function ==
			     AttachedToFunction::CudaMemcpyToSymbolAsync;
		if (async) {
			stream = (cudaStream_t)
				gum_invocation_context_get_nth_argument(gum_ctx,
									5);
		}
		context->impl->mirror_cuda_memcpy_to_symbol(
			symbol, src, count, offset, kind, stream, async);
	}
}

static void example_listener_on_leave(GumInvocationListener *listener,
				      GumInvocationContext *ic)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto context =
		GUM_IC_GET_FUNC_DATA(ic, CUDARuntimeFunctionHookerContext *);
	if (context->to_function == AttachedToFunction::RegisterFatbin) {
		SPDLOG_DEBUG("Leaving RegisterFatbin");
	} else if (context->to_function ==
		   AttachedToFunction::RegisterFunction) {
		SPDLOG_DEBUG("Leaving RegisterFunction");
	} else if (context->to_function ==
		   AttachedToFunction::RegisterVariable) {
		SPDLOG_DEBUG("Leaving __cudaRegisterVar");
	} else if (context->to_function ==
		   AttachedToFunction::RegisterFatbinEnd) {
		SPDLOG_DEBUG("Leaving __cudaRegisterFatBinaryEnd..");
	}
}

static void
cuda_runtime_function_hooker_class_init(CUDARuntimeFunctionHookerClass *klass)
{
}

static void cuda_runtime_function_hooker_iface_init(gpointer g_iface,
						    gpointer iface_data)
{
	auto iface = (GumInvocationListenerInterface *)g_iface;

	iface->on_enter = example_listener_on_enter;
	iface->on_leave = example_listener_on_leave;
}

static void cuda_runtime_function_hooker_init(CUDARuntimeFunctionHooker *self)
{
}

extern "C" cudaError_t
cuda_runtime_function__cudaLaunchKernel(const void *func, dim3 grid_dim,
					dim3 block_dim, void **args,
					size_t shared_mem, cudaStream_t stream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl != nullptr) {
		SPDLOG_DEBUG("grid_dim: {}, {}, {}", grid_dim.x, grid_dim.y,
			     grid_dim.z);
		SPDLOG_DEBUG("block_dim: {}, {}, {}", block_dim.x, block_dim.y,
			     block_dim.z);
	}
	return cuda_launch_kernel_common(impl,
					 impl->original_cuda_launch_kernel,
					 func, grid_dim, block_dim, args,
					 shared_mem, stream);
}

static std::optional<std::string>
cuda_graph_maybe_get_kernel_name_from_cufunction(nv_attach_impl &impl,
						 CUfunction function)
{
	if (auto cached = impl.find_original_kernel_name(function); cached)
		return cached;
	using cu_func_get_name_fn_t = CUresult (*)(const char **, CUfunction);
	static cu_func_get_name_fn_t cu_func_get_name =
		(cu_func_get_name_fn_t)dlsym(RTLD_DEFAULT, "cuFuncGetName");
	if (!cu_func_get_name)
		return std::nullopt;
	const char *name = nullptr;
	if (auto err = cu_func_get_name(&name, function); err != CUDA_SUCCESS)
		return std::nullopt;
	if (name == nullptr || name[0] == '\0')
		return std::nullopt;
	impl.record_original_cufunction_name(function, std::string(name));
	return std::string(name);
}

static std::optional<std::string>
cuda_graph_maybe_get_kernel_name_from_cukernel(CUkernel kernel)
{
	using cu_kernel_get_name_fn_t = CUresult (*)(const char **, CUkernel);
	static cu_kernel_get_name_fn_t cu_kernel_get_name =
		(cu_kernel_get_name_fn_t)dlsym(RTLD_DEFAULT, "cuKernelGetName");
	if (!cu_kernel_get_name)
		return std::nullopt;
	const char *name = nullptr;
	if (auto err = cu_kernel_get_name(&name, kernel); err != CUDA_SUCCESS)
		return std::nullopt;
	if (name == nullptr)
		return std::nullopt;
	return std::string(name);
}

extern "C" cudaError_t cuda_runtime_function__cudaLaunchKernel_ptsz(
	const void *func, dim3 grid_dim, dim3 block_dim, void **args,
	size_t shared_mem, cudaStream_t stream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl != nullptr) {
		SPDLOG_DEBUG("grid_dim: {}, {}, {}", grid_dim.x, grid_dim.y,
			     grid_dim.z);
		SPDLOG_DEBUG("block_dim: {}, {}, {}", block_dim.x, block_dim.y,
			     block_dim.z);
	}
	return cuda_launch_kernel_common(impl,
					 impl->original_cuda_launch_kernel_ptsz,
					 func, grid_dim, block_dim, args,
					 shared_mem, stream);
}

static const CUDA_KERNEL_NODE_PARAMS_v1 *
cuda_graph_maybe_patch_kernel_node_params_v1(
	nv_attach_impl &impl, const CUDA_KERNEL_NODE_PARAMS_v1 *params,
	CUDA_KERNEL_NODE_PARAMS_v1 &patched_params)
{
	if (params == nullptr || params->func == nullptr) {
		return params;
	}
	auto kernel_name = cuda_graph_maybe_get_kernel_name_from_cufunction(
		impl, params->func);
	if (!kernel_name) {
		return params;
	}
	auto patched_func = impl.find_patched_kernel_function(*kernel_name);
	if (!patched_func) {
		return params;
	}
	patched_params = *params;
	patched_params.func = *patched_func;
	return &patched_params;
}

static const CUDA_KERNEL_NODE_PARAMS_v2 *
cuda_graph_maybe_patch_kernel_node_params_v2(
	nv_attach_impl &impl, const CUDA_KERNEL_NODE_PARAMS_v2 *params,
	CUDA_KERNEL_NODE_PARAMS_v2 &patched_params)
{
	if (params == nullptr) {
		return params;
	}
	std::optional<std::string> kernel_name;
	if (params->func != nullptr) {
		kernel_name = cuda_graph_maybe_get_kernel_name_from_cufunction(
			impl, params->func);
	} else if (params->kern != nullptr) {
		kernel_name = cuda_graph_maybe_get_kernel_name_from_cukernel(
			params->kern);
	} else {
	}
	if (!kernel_name) {
		return params;
	}
	auto patched_func = impl.find_patched_kernel_function(*kernel_name);
	if (!patched_func) {
		return params;
	}
	patched_params = *params;
	patched_params.func = *patched_func;
	patched_params.kern = nullptr;
	return &patched_params;
}

extern "C" CUresult cuda_driver_function__cuGraphAddKernelNode_v1(
	CUgraphNode *phGraphNode, CUgraph hGraph,
	const CUgraphNode *dependencies, size_t numDependencies,
	const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl == nullptr)
		return CUDA_ERROR_UNKNOWN;
	auto original = reinterpret_cast<cu_graph_add_kernel_node_v1_fn_t>(
		impl->original_cu_graph_add_kernel_node_v1);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	CUDA_KERNEL_NODE_PARAMS_v1 patched_params;
	auto params_to_use = cuda_graph_maybe_patch_kernel_node_params_v1(
		*impl, nodeParams, patched_params);
	return original(phGraphNode, hGraph, dependencies, numDependencies,
			params_to_use);
}

extern "C" CUresult cuda_driver_function__cuGraphAddKernelNode_v2(
	CUgraphNode *phGraphNode, CUgraph hGraph,
	const CUgraphNode *dependencies, size_t numDependencies,
	const CUDA_KERNEL_NODE_PARAMS_v2 *nodeParams)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl == nullptr)
		return CUDA_ERROR_UNKNOWN;
	auto original = reinterpret_cast<cu_graph_add_kernel_node_v2_fn_t>(
		impl->original_cu_graph_add_kernel_node_v2);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	CUDA_KERNEL_NODE_PARAMS_v2 patched_params;
	auto params_to_use = cuda_graph_maybe_patch_kernel_node_params_v2(
		*impl, nodeParams, patched_params);
	return original(phGraphNode, hGraph, dependencies, numDependencies,
			params_to_use);
}

extern "C" CUresult cuda_driver_function__cuGraphExecKernelNodeSetParams_v1(
	CUgraphExec hGraphExec, CUgraphNode hNode,
	const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl == nullptr)
		return CUDA_ERROR_UNKNOWN;
	auto original =
		reinterpret_cast<cu_graph_exec_kernel_node_set_params_v1_fn_t>(
			impl->original_cu_graph_exec_kernel_node_set_params_v1);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	CUDA_KERNEL_NODE_PARAMS_v1 patched_params;
	auto params_to_use = cuda_graph_maybe_patch_kernel_node_params_v1(
		*impl, nodeParams, patched_params);
	return original(hGraphExec, hNode, params_to_use);
}

extern "C" CUresult cuda_driver_function__cuGraphExecKernelNodeSetParams_v2(
	CUgraphExec hGraphExec, CUgraphNode hNode,
	const CUDA_KERNEL_NODE_PARAMS_v2 *nodeParams)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl == nullptr)
		return CUDA_ERROR_UNKNOWN;
	auto original =
		reinterpret_cast<cu_graph_exec_kernel_node_set_params_v2_fn_t>(
			impl->original_cu_graph_exec_kernel_node_set_params_v2);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	CUDA_KERNEL_NODE_PARAMS_v2 patched_params;
	auto params_to_use = cuda_graph_maybe_patch_kernel_node_params_v2(
		*impl, nodeParams, patched_params);
	return original(hGraphExec, hNode, params_to_use);
}

extern "C" CUresult cuda_driver_function__cuGraphKernelNodeSetParams_v1(
	CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl == nullptr)
		return CUDA_ERROR_UNKNOWN;
	auto original =
		reinterpret_cast<cu_graph_kernel_node_set_params_v1_fn_t>(
			impl->original_cu_graph_kernel_node_set_params_v1);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	CUDA_KERNEL_NODE_PARAMS_v1 patched_params;
	auto params_to_use = cuda_graph_maybe_patch_kernel_node_params_v1(
		*impl, nodeParams, patched_params);
	return original(hNode, params_to_use);
}

extern "C" CUresult cuda_driver_function__cuGraphKernelNodeSetParams_v2(
	CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v2 *nodeParams)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl == nullptr)
		return CUDA_ERROR_UNKNOWN;
	auto original =
		reinterpret_cast<cu_graph_kernel_node_set_params_v2_fn_t>(
			impl->original_cu_graph_kernel_node_set_params_v2);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	CUDA_KERNEL_NODE_PARAMS_v2 patched_params;
	auto params_to_use = cuda_graph_maybe_patch_kernel_node_params_v2(
		*impl, nodeParams, patched_params);
	return original(hNode, params_to_use);
}
