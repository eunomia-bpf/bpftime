// #include "pos/hip_impl/utils/fatbin.h"
#include "spdlog/common.h"
#include "spdlog/spdlog.h"
#include <cassert>
#include <cstdint>
#include <frida-gum.h>
#include "rocm_attach_impl.hpp"
#include <clang/Driver/OffloadBundler.h>
#include <llvm/ADT/StringRef.h>
#include <vector>
#include <fstream>
using namespace bpftime;
using namespace attach;
constexpr char kOffloadBundleUncompressedMagicStr[] =
	"__CLANG_OFFLOAD_BUNDLE__";
static constexpr size_t kOffloadBundleUncompressedMagicStrSize =
	sizeof(kOffloadBundleUncompressedMagicStr);

struct __ClangOffloadBundleInfo {
	uint64_t offset;
	uint64_t size;
	uint64_t bundleEntryIdSize;
	const char bundleEntryId[1];
};

struct __ClangOffloadBundleUncompressedHeader {
	const char magic[kOffloadBundleUncompressedMagicStrSize - 1];
	uint64_t numOfCodeObjects;
	__ClangOffloadBundleInfo desc[1];
};

struct HipFatbinWrapper {
	// should be 1212764230
	uint32_t magic;
	// 1
	uint32_t unknown_field_1;
	// pointer to clang offload bundle
	__ClangOffloadBundleUncompressedHeader *data;
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
		std::vector<std::string> hip_code;
		auto start = (const char *)&arg1->data->desc;
		auto ptr = start;
		for (uint64_t i = 0; i < arg1->data->numOfCodeObjects; i++) {
			auto desc = (__ClangOffloadBundleInfo *)ptr;
			std::string curr_arch(
				&desc->bundleEntryId[0],
				&desc->bundleEntryId[desc->bundleEntryIdSize]);
			SPDLOG_INFO("Checking arch {}, offset = {}", curr_arch,
				    desc->offset);
			if (curr_arch.starts_with("hipv4-amdgcn-amd")) {
				SPDLOG_INFO("Saving code..");
				hip_code.emplace_back(start + desc->offset,
						      start + desc->size);
				if (spdlog::should_log(spdlog::level::debug)) {
					std::string path = "/tmp/out.txt";
					SPDLOG_DEBUG("Saving code to {}", path);
					std::ofstream ofs(path);
					ofs << hip_code.back();
				}
			} else {
				SPDLOG_INFO("Ignored this arch");
			}
			ptr = &desc->bundleEntryId[0] + desc->bundleEntryIdSize;
		}
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
