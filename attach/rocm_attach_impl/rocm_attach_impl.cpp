#include "rocm_attach_impl.hpp"
#include "frida-gum.h"
#include "rocm_attach_private_data.hpp"
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
	if (attach_type == ATTACH_ROCM_PROBE_AND_RETPROBE) {
		int id = this->allocate_id();
		const auto &data =
			dynamic_cast<const rocm_attach_private_data &>(
				private_data);
		this->hook_entries[id] = rocm_attach_entry{
			.type =
				rocm_attach_function_probe{
					.func = data.func_name,
					.is_retprobe = data.is_ret_probe },
			.instructions = data.instructions
		};
		this->map_basic_info = data.map_basic_info;
		this->shared_mem = data.comm_shared_mem;
		SPDLOG_INFO(
			"Recording probe/retprobe for rocm: func_name={}, retprobe={}, insn count={}, shared_mem_ptr={:x}",
			data.func_name, data.is_ret_probe,
			data.instructions.size(), data.comm_shared_mem);
		return id;
	} else {
		SPDLOG_ERROR("Unsupported attach type by rocm attach impl: {}",
			     attach_type);
		return -1;
	}
}
