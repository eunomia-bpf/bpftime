// #include "pos/cuda_impl/utils/fatbin.h"
#include "spdlog/spdlog.h"
#include <cassert>
#include <cstdint>
#include <frida-gum.h>
#include <vector>
#include "nv_attach_impl.hpp"
#include <dlfcn.h>

using namespace bpftime;
using namespace attach;

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
				SPDLOG_INFO(
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
		auto patched_fatbin =
			context->impl->hack_fatbin(std::move(data_vec)).value();
		auto patched_fatbin_ptr =
			std::make_unique<std::vector<uint8_t>>(patched_fatbin);
		auto patched_header = std::make_unique<__fatBinC_Wrapper_t>();
		auto patched_header_ptr = patched_header.get();
		patched_header->magic = 0x466243b1;
		patched_header->version = 1;
		patched_header->data =
			(const unsigned long long *)patched_fatbin_ptr->data();
		patched_header->filename_or_fatbins = 0;
		context->impl->stored_binaries_body.push_back(
			std::move(patched_fatbin_ptr));
		context->impl->stored_binaries_header.push_back(
			std::move(patched_header));
		// 不替换原 fatbin，保留应用自身符号
	} else if (context->to_function ==
		   AttachedToFunction::RegisterFunction) {
		SPDLOG_DEBUG("Entering __cudaRegisterFunction..");

	} else if (context->to_function ==
		   AttachedToFunction::RegisterFatbinEnd) {
		SPDLOG_DEBUG("Entering __cudaRegisterFatBinaryEnd..");
		auto &impl = *context->impl;
		// 在 FatbinEnd 时并行注册我们自己的已打补丁 fatbin
		if (!impl.stored_binaries_header.empty()) {
			auto patched_header_ptr =
				impl.stored_binaries_header.back().get();
			using RegFatbinFn = void **(*)(__fatBinC_Wrapper_t *);
			auto reg_sym =
				dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
			if (!reg_sym) {
				SPDLOG_ERROR(
					"dlsym __cudaRegisterFatBinary failed");
				assert(false);
			}
			auto reg_f = reinterpret_cast<RegFatbinFn>(reg_sym);
			void **new_handle = reg_f(patched_header_ptr);
			impl.stored_binaries_handles.push_back(new_handle);
			if (impl.trampoline_memory_state ==
			    TrampolineMemorySetupStage::NotSet) {
				if (int err = impl.register_trampoline_memory(
					    new_handle);
				    err != 0) {
					assert(false);
				}
			}
			// 调用 __cudaRegisterFatBinaryEnd(new_handle)
			using RegFatbinEndFn = void (*)(void **);
			auto end_sym =
				dlsym(RTLD_NEXT, "__cudaRegisterFatBinaryEnd");
			if (!end_sym) {
				SPDLOG_ERROR(
					"dlsym __cudaRegisterFatBinaryEnd failed");
				assert(false);
			}
			auto end_f = reinterpret_cast<RegFatbinEndFn>(end_sym);
			end_f(new_handle);
		}
	} else if (context->to_function ==
		   AttachedToFunction::CudaLaunchKernel) {
		SPDLOG_DEBUG("Entering cudaLaunchKernel");
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
		   AttachedToFunction::RegisterFatbinEnd) {
		SPDLOG_DEBUG("Leaving __cudaRegisterFatBinaryEnd..");
		if (int err = context->impl->copy_data_to_trampoline_memory();
		    err != 0) {
			SPDLOG_ERROR("Unable to copy data to trampoline");
			assert(false);
		}
	} else if (context->to_function ==
		   AttachedToFunction::CudaLaunchKernel) {
		SPDLOG_DEBUG("Leaving cudaLaunchKernel");
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
