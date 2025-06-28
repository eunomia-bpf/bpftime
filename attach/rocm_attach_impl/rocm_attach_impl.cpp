#include "rocm_attach_impl.hpp"
#include "frida-gum.h"
#include "spdlog/spdlog.h"
#include <dlfcn.h>
using namespace bpftime;
using namespace attach;
extern GType rocm_runtime_function_hooker_get_type();
rocm_attach_impl::rocm_attach_impl()
{
	SPDLOG_INFO("Initializing rocm attach impl");
	gum_init_embedded();
	auto interceptor = gum_interceptor_obtain();
	assert(interceptor != nullptr);
	auto listener =
		g_object_new(rocm_runtime_function_hooker_get_type(), nullptr);
	assert(listener != nullptr);
	this->frida_interceptor = interceptor;
	this->frida_listener = listener;
	gum_interceptor_begin_transaction(interceptor);
	auto register_hook = [&](RocmAttachedToFunction func, void *addr) {
		auto ctx = std::make_unique<ROCMRuntimeFunctionHookerContext>();
		ctx->to_function = func;
		ctx->impl = this;
		auto ctx_ptr = ctx.get();
		this->hooker_contexts.push_back(std::move(ctx));
		if (auto result = gum_interceptor_attach(
			    interceptor, (gpointer)addr,
			    (GumInvocationListener *)listener, ctx_ptr);
		    result != GUM_ATTACH_OK) {
			SPDLOG_ERROR("Unable to attach to CUDA functions: {}",
				     (int)result);
			assert(false);
		}
	};
	register_hook(RocmAttachedToFunction::RegisterFatbin,
		      (gpointer)dlsym(RTLD_NEXT, "__hipRegisterFatBinary"));
	register_hook(RocmAttachedToFunction::RegisterFunction,
		      GSIZE_TO_POINTER(gum_module_find_export_by_name(
			      nullptr, "__hipRegisterFunction")));
	register_hook(RocmAttachedToFunction::HipLaunchKernel,
		      GSIZE_TO_POINTER(gum_module_find_export_by_name(
			      nullptr, "hipLaunchKernel")));

	gum_interceptor_end_transaction(interceptor);
}

rocm_attach_impl::~rocm_attach_impl()
{
	if (frida_listener)
		g_object_unref(frida_listener);
}

int rocm_attach_impl::detach_by_id(int id)
{
	SPDLOG_WARN("Detaching is not supported by rocm attach impl yet");
	return -1;
}

int rocm_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&cb, const attach_private_data &private_data,
	int attach_type)
{
	SPDLOG_WARN("TODO: rocm_attach_impl::create_attach_with_ebpf_callback");
	return -1;
}
