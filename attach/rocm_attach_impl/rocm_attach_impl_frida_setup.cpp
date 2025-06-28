// #include "pos/hip_impl/utils/fatbin.h"
#include "spdlog/spdlog.h"
#include <cassert>
#include <cstdint>
#include <frida-gum.h>
#include "rocm_attach_impl.hpp"
#include <clang/Driver/OffloadBundler.h>
#include <llvm/ADT/StringRef.h>
#include "llvm/Object/OffloadBinary.h"
using namespace bpftime;
using namespace attach;

struct HipFatbinWrapper {
	// should be 1212764230
	uint32_t magic;
	// 1
	uint32_t unknown_field_1;
	// pointer to clang offload bundle
	void *data;
	// 0
	uint64_t unknown_field_2;
};

typedef struct _ROCMRuntimeFunctionHooker {
	GObject parent;
} ROCMRuntimeFunctionHooker;

static void rocm_runtime_function_hooker_iface_init(gpointer g_iface,
						    gpointer iface_data);

G_DECLARE_FINAL_TYPE(ROCMRuntimeFunctionHooker, rocm_runtime_function_hooker,
		     BPFTIME, ROCM_ATTACH_IMPL, GObject)
G_DEFINE_TYPE_EXTENDED(
	ROCMRuntimeFunctionHooker, rocm_runtime_function_hooker, G_TYPE_OBJECT,
	0,
	G_IMPLEMENT_INTERFACE(GUM_TYPE_INVOCATION_LISTENER,
			      rocm_runtime_function_hooker_iface_init))

static void rocm_listener_on_enter(GumInvocationListener *listener,
				   GumInvocationContext *ic)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto context =
		GUM_IC_GET_FUNC_DATA(ic, ROCMRuntimeFunctionHookerContext *);
	if (context->to_function == RocmAttachedToFunction::RegisterFatbin) {
		SPDLOG_DEBUG("Entering __hipRegisterFatbin");
		auto arg1 = (HipFatbinWrapper *)
			gum_invocation_context_get_nth_argument(gum_ctx, 0);
		std::unique_ptr<llvm::MemoryBuffer> input_buffer =
			llvm::MemoryBuffer::getMemBuffer(
				llvm::StringRef((const char *)arg1->data),
				"bundled_input");
		auto file = llvm::object::OffloadBinary::create(
			std::move(*input_buffer));
			
	} else if (context->to_function ==
		   RocmAttachedToFunction::RegisterFunction) {
		SPDLOG_DEBUG("Entering __hipRegisterFunction..");

	} else if (context->to_function ==
		   RocmAttachedToFunction::RegisterFatbinEnd) {
		SPDLOG_DEBUG("Entering __hipRegisterFatBinaryEnd..");

	} else if (context->to_function ==
		   RocmAttachedToFunction::HipLaunchKernel) {
		SPDLOG_DEBUG("Entering hipLaunchKernel");
	}
}

static void rocm_listener_on_leave(GumInvocationListener *listener,
				   GumInvocationContext *ic)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto context =
		GUM_IC_GET_FUNC_DATA(ic, ROCMRuntimeFunctionHookerContext *);
	if (context->to_function == RocmAttachedToFunction::RegisterFatbin) {
		SPDLOG_DEBUG("Leaving RegisterFatbin");
	} else if (context->to_function ==
		   RocmAttachedToFunction::RegisterFunction) {
		SPDLOG_DEBUG("Leaving RegisterFunction");
	} else if (context->to_function ==
		   RocmAttachedToFunction::RegisterFatbinEnd) {
	} else if (context->to_function ==
		   RocmAttachedToFunction::HipLaunchKernel) {
		SPDLOG_DEBUG("Leaving hipLaunchKernel");
	}
}

static void
rocm_runtime_function_hooker_class_init(ROCMRuntimeFunctionHookerClass *klass)
{
}

static void rocm_runtime_function_hooker_iface_init(gpointer g_iface,
						    gpointer iface_data)
{
	auto iface = (GumInvocationListenerInterface *)g_iface;

	iface->on_enter = rocm_listener_on_enter;
	iface->on_leave = rocm_listener_on_leave;
}

static void rocm_runtime_function_hooker_init(ROCMRuntimeFunctionHooker *self)
{
}
