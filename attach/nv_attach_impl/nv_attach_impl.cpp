#include "nv_attach_impl.hpp"
#include "frida-gum.h"

#include "pos/include/common.h"
#include "spdlog/spdlog.h"
#include <asm/unistd.h> // For architecture-specific syscall numbers

#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <memory>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <sys/uio.h>
#include <link.h>
#include <pos/cuda_impl/utils/fatbin.h>
using namespace bpftime;
using namespace attach;

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

extern "C" {
typedef struct {
	int magic;
	int version;
	const unsigned long long *data;
	void *filename_or_fatbins;

} __fatBinC_Wrapper_t;
typedef struct __attribute__((__packed__)) fat_elf_header {
	uint32_t magic;
	uint16_t version;
	uint16_t header_size;
	uint64_t size;
} fat_elf_header_t;
}

static void example_listener_on_enter(GumInvocationListener *listener,
				      GumInvocationContext *ic)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto context =
		GUM_IC_GET_FUNC_DATA(ic, CUDARuntimeFunctionHookerContext *);
	if (context->to_function == AttachedToFunction::RegisterFatbin) {
		SPDLOG_INFO("Mocking __cudaRegisterFatBinary..");

		auto header = (__fatBinC_Wrapper_t *)
			gum_invocation_context_get_nth_argument(gum_ctx, 0);
		auto data = (const char *)header->data;
		fat_elf_header_t *curr_header = (fat_elf_header_t *)data;
		const char *tail = (const char *)curr_header;
		while (true) {
			if (curr_header->magic == FATBIN_TEXT_MAGIC) {
				SPDLOG_INFO(
					"Got CUBIN section header size = {}, size = {}",
					curr_header->header_size,
					curr_header->size);
				tail = ((const char *)curr_header) +
				       curr_header->header_size +
				       curr_header->size;
				curr_header = (fat_elf_header_t *)tail;
			} else {
				break;
			}
		};
		std::vector<char> data_vec(data, tail);
		SPDLOG_INFO("Finally size = {}", data_vec.size());

		std::vector<POSCudaFunctionDesp *> desp;
		std::map<std::string, POSCudaFunctionDesp *> cache;

		auto result =
			POSUtil_CUDA_Fatbin::obtain_functions_from_cuda_binary(
				(uint8_t *)data_vec.data(), data_vec.size(),
				&desp, cache);
		if (result != POS_SUCCESS) {
			SPDLOG_ERROR(
				"Unable to parse functions from fatbin: {}",
				(int)result);
			return;
		}
		SPDLOG_INFO("Got these functions in the fatbin");
		for (const auto &item : desp) {
			SPDLOG_INFO("{}", item->signature);
		}
	}
}

static void example_listener_on_leave(GumInvocationListener *listener,
				      GumInvocationContext *ic)
{
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

int nv_attach_impl::detach_by_id(int id)
{
	return 0;
}

int nv_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&cb, const attach_private_data &private_data,
	int attach_type)
{
	return 0;
}
nv_attach_impl::nv_attach_impl()
{
	SPDLOG_INFO("Starting nv_attach_impl");
	gum_init_embedded();
	auto interceptor = gum_interceptor_obtain();
	assert(interceptor != nullptr);
	auto listener =
		g_object_new(cuda_runtime_function_hooker_get_type(), nullptr);
	assert(listener != nullptr);
	this->frida_interceptor = interceptor;
	this->frida_listener = listener;

	gum_interceptor_begin_transaction(interceptor);
	auto ctx = std::make_unique<CUDARuntimeFunctionHookerContext>();
	ctx->to_function = AttachedToFunction::RegisterFatbin;
	ctx->impl = this;
	auto ctx_ptr = ctx.get();
	this->hooker_contexts.push_back(std::move(ctx));
	auto original___cudaRegisterFatBinary =
		(void **(*)(void *))dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
	auto result = gum_interceptor_attach(
		interceptor,
		// GSIZE_TO_POINTER(gum_module_find_export_by_name(
		// 	nullptr, "__cudaRegisterFatBinary"))
		(gpointer)original___cudaRegisterFatBinary

		,
		(GumInvocationListener *)listener, ctx_ptr);
	if (result != GUM_ATTACH_OK) {
		SPDLOG_ERROR("Unable to attach to CUDA functions: {}",
			     (int)result);
		assert(false);
	}
	gum_interceptor_end_transaction(interceptor);
	SPDLOG_INFO("Probe to __cudaRegisterFatBinary attached");
}

nv_attach_impl::~nv_attach_impl()
{
	if (frida_listener)
		g_object_unref(frida_listener);
}
// static void **(*original___cudaRegisterFatBinary)(void *) = nullptr;

// extern "C" void **__cudaRegisterFatBinary(void *fatbin)
// {
// 	SPDLOG_INFO("Mocking __cudaRegisterFatBinary..");

// 	auto orig = try_get_original_func("__cudaRegisterFatBinary",
// 					  original___cudaRegisterFatBinary);

// 	auto header = (__fatBinC_Wrapper_t *)fatbin;
// 	auto data = (const char *)header->data;
// 	fat_elf_header_t *curr_header = (fat_elf_header_t *)data;
// 	const char *tail = (const char *)curr_header;
// 	while (true) {
// 		if (curr_header->magic == FATBIN_TEXT_MAGIC) {
// 			SPDLOG_INFO(
// 				"Got CUBIN section header size = {}, size = {}",
// 				curr_header->header_size, curr_header->size);
// 			tail = ((const char *)curr_header) +
// 			       curr_header->header_size + curr_header->size;
// 			curr_header = (fat_elf_header_t *)tail;
// 		} else {
// 			break;
// 		}
// 	};
// 	std::vector<char> data_vec(data, tail);
// 	SPDLOG_INFO("Finally size = {}", data_vec.size());

// 	std::vector<POSCudaFunctionDesp *> desp;
// 	std::map<std::string, POSCudaFunctionDesp *> cache;

// 	auto result = POSUtil_CUDA_Fatbin::obtain_functions_from_cuda_binary(
// 		(uint8_t *)data_vec.data(), data_vec.size(), &desp, cache);
// 	if (result != POS_SUCCESS) {
// 		SPDLOG_ERROR("Unable to parse functions from fatbin: {}",
// 			     (int)result);
// 		return orig(fatbin);
// 	}
// 	SPDLOG_INFO("Got these functions in the fatbin");
// 	for (const auto &item : desp) {
// 		SPDLOG_INFO("{}, {}", item->name, item->signature);
// 	}
// 	return orig(fatbin);
// }
