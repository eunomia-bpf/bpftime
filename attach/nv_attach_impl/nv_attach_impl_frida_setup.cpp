// #include "pos/cuda_impl/utils/fatbin.h"
#include "cuda.h"
#include "driver_types.h"
#include "spdlog/spdlog.h"
#include "vector_types.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <frida-gum.h>
#include <iterator>
#include <memory>
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
		// auto patched =
		// context->impl->hack_fatbin(std::move(data_vec)); if
		// (!patched.has_value()) { 	SPDLOG_ERROR(
		// "hack_fatbin failed; skipping fatbin replacement"); 	return;
		// }
		// auto &patched_fatbin = *patched;
		auto extracted_ptx =
			context->impl->extract_ptxs(std::move(data_vec));
		SPDLOG_INFO("Patching PTXs");
		auto patched_ptx = *context->impl->hack_fatbin(extracted_ptx);
		auto fatbin_record = std::make_unique<struct fatbin_record>();
		fatbin_record->fatbin_handle = (void **)header;
		auto &impl = *context->impl;
		for (const auto &[name, ptx] : patched_ptx) {
			CUmodule module;
			SPDLOG_INFO("Loading module: {}", name);
			char error_buf[8192], info_buf[8192];
			CUjit_option options[] = {
				CU_JIT_INFO_LOG_BUFFER,
				CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
				CU_JIT_ERROR_LOG_BUFFER,
				CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
			};
			void *option_values[] = {
				(void *)info_buf, (void *)std::size(info_buf),
				(void *)error_buf, (void *)std::size(error_buf)
			};
			if (auto err = cuModuleLoadDataEx(
				    &module, ptx.data(), std::size(options),
				    options, option_values);
			    err != CUDA_SUCCESS) {
				SPDLOG_ERROR("Unable to compile module {}: {}",
					     name, (int)err);
				SPDLOG_ERROR("Info: {}", info_buf);
				SPDLOG_ERROR("Error: {}", error_buf);
				throw std::runtime_error(
					"Unable to compile module");
			}
			CUdeviceptr const_data_ptr, map_basic_info_ptr;
			size_t const_data_size, map_basic_info_size;
			SPDLOG_INFO("Copying trampoline data to device");
			CUDA_DRIVER_CHECK_EXCEPTION(
				cuModuleGetGlobal(&const_data_ptr,
						  &const_data_size, module,
						  "constData"),
				"Unable to get pointer of constData");
			CUDA_DRIVER_CHECK_EXCEPTION(
				cuModuleGetGlobal(&map_basic_info_ptr,
						  &map_basic_info_size, module,
						  "map_info"),
				"Unable to get pointer of map_info");
			CUDA_DRIVER_CHECK_EXCEPTION(
				cuMemcpyHtoD(const_data_ptr,
					     &context->impl->shared_mem_ptr,
					     const_data_size),
				"Unable to copy constData pointer to device");
			CUDA_DRIVER_CHECK_EXCEPTION(
				cuMemcpyHtoD(
					map_basic_info_ptr,
					context->impl->map_basic_info->data(),
					map_basic_info_size),
				"Unable to copy constData pointer to device");
			SPDLOG_INFO("Trampoline data copied");
			fatbin_record->ptxs.emplace_back(
				std::make_unique<fatbin_record::ptx_in_module>(
					module));
			SPDLOG_INFO("Loaded module: {}", name);
		}
		impl.current_fatbin = fatbin_record.get();
		impl.fatbin_records.emplace_back(std::move(fatbin_record));

		// auto patched_fatbin_ptr =
		// 	std::make_unique<std::vector<uint8_t>>(patched_fatbin);
		// auto patched_header =
		// std::make_unique<__fatBinC_Wrapper_t>(); auto
		// patched_header_ptr = patched_header.get();
		// patched_header->magic = 0x466243b1;
		// patched_header->version = 1;
		// patched_header->data =
		// 	(const unsigned long long *)patched_fatbin_ptr->data();
		// patched_header->filename_or_fatbins = 0;
		// context->impl->stored_binaries_body.push_back(
		// 	std::move(patched_fatbin_ptr));
		// context->impl->stored_binaries_header.push_back(
		// 	std::move(patched_header));
		// // Set the patched header as the argument
		// gum_invocation_context_replace_nth_argument(gum_ctx, 0,
		// 					    patched_header_ptr);
	} else if (context->to_function ==
		   AttachedToFunction::RegisterFunction) {
		SPDLOG_DEBUG("Entering __cudaRegisterFunction..");
		auto fatbin_handle =
			gum_invocation_context_get_nth_argument(gum_ctx, 0);
		auto current_fatbin = context->impl->current_fatbin;

		auto func_addr =
			gum_invocation_context_get_nth_argument(gum_ctx, 1);
		auto symbol_name =
			(const char *)gum_invocation_context_get_nth_argument(
				gum_ctx, 3);
		if (fatbin_handle != current_fatbin->fatbin_handle) {
			SPDLOG_DEBUG(
				"When registering kernel function {}, the provided fatbin handle (0x{:x}) doesn't match the fatbin handle provided by cudaRegisterFatbin (0x{:x})",
				symbol_name, (uintptr_t)fatbin_handle,
				(uintptr_t)current_fatbin->fatbin_handle);
		}
		if (auto result = current_fatbin->find_and_fill_function_info(
			    func_addr, symbol_name);
		    !result) {
			SPDLOG_ERROR("Unable to find func symbol named {}",
				     symbol_name);
		} else {
			context->impl->symbol_address_to_fatbin[func_addr] =
				current_fatbin;
			SPDLOG_DEBUG(
				"Registered kernel function name {} addr {:x}",
				symbol_name, (uintptr_t)func_addr);
		}

	} else if (context->to_function ==
		   AttachedToFunction::RegisterVariable) {
		SPDLOG_DEBUG("Entering __cudaRegisterVar");
		auto current_fatbin = context->impl->current_fatbin;

		auto fatbin_handle =
			gum_invocation_context_get_nth_argument(gum_ctx, 0);
		auto var_addr =
			gum_invocation_context_get_nth_argument(gum_ctx, 1);
		auto symbol_name =
			(const char *)gum_invocation_context_get_nth_argument(
				gum_ctx, 3);
		auto ext =
			(int)(uintptr_t)gum_invocation_context_get_nth_argument(
				gum_ctx, 4);
		auto size = (size_t)(uintptr_t)
			gum_invocation_context_get_nth_argument(gum_ctx, 5);

		auto constant =
			(int)(uintptr_t)gum_invocation_context_get_nth_argument(
				gum_ctx, 6);
		auto global =
			(int)(uintptr_t)gum_invocation_context_get_nth_argument(
				gum_ctx, 7);
		SPDLOG_DEBUG(
			"Registering variable named {}, ext={}, size={}, constant={}, global={}",
			symbol_name, ext, size, constant, global);
		if (fatbin_handle != current_fatbin->fatbin_handle) {
			SPDLOG_DEBUG(
				"When registering variable {}, the provided fatbin handle (0x{:x}) doesn't match the fatbin handle provided by cudaRegisterFatbin (0x{:x})",
				symbol_name, (uintptr_t)fatbin_handle,
				(uintptr_t)current_fatbin->fatbin_handle);
		}
		if (auto result = current_fatbin->find_and_fill_variable_info(
			    var_addr, symbol_name);
		    !result) {
			SPDLOG_ERROR("Unable to find variable symbol named {}",
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
		// auto &impl = *context->impl;
		// auto arg = (void **)
		// 		gum_invocation_context_get_nth_argument(gum_ctx,
		// 							0);
		// 	if (int err = impl.register_trampoline_memory(arg);
		// 	    err != 0) {
		// 		assert(false);
		// 	}
		auto &current_fatbin = context->impl->current_fatbin;
		SPDLOG_INFO("Got {} functions, {} variables defined",
			    current_fatbin->function_addr_to_symbol.size(),
			    current_fatbin->variable_addr_to_symbol.size());
		current_fatbin = nullptr;
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
		// if (int err =
		// context->impl->copy_data_to_trampoline_memory();
		//     err != 0) {
		// 	SPDLOG_ERROR(
		// 		"Unable to copy data to trampoline, skipping due
		// to environment");
		// 	// Do not abort process; continue without device-side
		// 	// trampolines
		// }
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
	SPDLOG_DEBUG("Try access: {}", impl->fatbin_records.size());
	SPDLOG_DEBUG("grid_dim: {}, {}, {}", grid_dim.x, grid_dim.y,
		     grid_dim.z);
	SPDLOG_DEBUG("block_dim: {}, {}, {}", block_dim.x, block_dim.y,
		     block_dim.z);
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
			SPDLOG_ERROR("Unable to launch kernel: {}", (int)err);
			return cudaErrorLaunchFailure;
		}
		return cudaSuccess;

	} else {
		SPDLOG_DEBUG("Symbol not found ");
		return cudaErrorSymbolNotFound;
	}
}
