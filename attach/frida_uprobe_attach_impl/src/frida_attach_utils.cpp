#include "frida_attach_utils.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include <filesystem>
#include <spdlog/spdlog.h>
#include <frida-gum.h>
static std::string get_executable_path()
{
	char exec_path[PATH_MAX] = { 0 };
	ssize_t len =
		readlink("/proc/self/exe", exec_path, sizeof(exec_path) - 1);
	if (len != -1) {
		exec_path[len] = '\0'; // Null-terminate the string
		SPDLOG_INFO("Executable path: {}", exec_path);
	} else {
		SPDLOG_ERROR("Error retrieving executable path: {}", errno);
	}
	return exec_path;
}
namespace bpftime
{
namespace attach
{
void *
resolve_function_addr_by_module_offset(const std::string_view &module_name,
				       uintptr_t func_offset)
{
	auto exec_path = get_executable_path();
	void *module_base_addr = nullptr;
	SPDLOG_DEBUG("Resolving module base addr, module name {}, exec_path {}",
		     module_name, exec_path);
	if (std::filesystem::equivalent(module_name, exec_path)) {
		SPDLOG_DEBUG(
			"module name {} is equivalent to exec path {}, using empty string to resolve module base addr",
			module_name, exec_path);
		module_base_addr = get_module_base_addr("");
	} else {
			SPDLOG_DEBUG(
			"module name {} is *not* equivalent to exec path {}, using module name to resolve module base addr",
			module_name, exec_path);
		module_base_addr =
			get_module_base_addr(std::string(module_name).c_str());
	}
	if (!module_base_addr) {
		SPDLOG_ERROR(
			"Failed to find module base address for {}, cwd = {}",
			module_name, std::filesystem::current_path().c_str());
		return nullptr;
	}

	return ((char *)module_base_addr) + func_offset;
}

void *find_function_addr_by_name(const char *name)
{
	if (auto ptr = gum_find_function(name); ptr)
		return ptr;
	if (auto ptr = (void *)gum_module_find_export_by_name(nullptr, name);
	    ptr)
		return ptr;
	return nullptr;
}

void *get_module_base_addr(const char *module_name)
{
	gum_module_load(module_name, nullptr);
	return (void *)gum_module_find_base_address(module_name);
}
void *find_module_export_by_name(const char *module_name,
				 const char *symbol_name)
{
	return (void *)(uintptr_t)gum_module_find_export_by_name(module_name,
								 symbol_name);
}
int from_cb_idx_to_attach_type(int idx)
{
	switch (idx) {
	case ATTACH_UPROBE_INDEX:
		return ATTACH_UPROBE;
	case ATTACH_UPROBE_OVERRIDE_INDEX:
		return ATTACH_UPROBE_OVERRIDE;
	case ATTACH_URETPROBE_INDEX:
		return ATTACH_URETPROBE;
	default:
		assert(false && "Unreachable!");
	}
	return 0;
}
} // namespace attach
} // namespace bpftime

extern "C" uint64_t bpftime_get_func_ret(uint64_t ctx, uint64_t *value,
					 uint64_t, uint64_t, uint64_t)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EOPNOTSUPP;
	}
	// ignore ctx;
	*value = (uint64_t)gum_invocation_context_get_return_value(gum_ctx);
	return 0;
}

extern "C" uint64_t bpftime_get_func_arg(uint64_t ctx, uint32_t n,
					 uint64_t *value, uint64_t, uint64_t)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EINVAL;
	}
	// ignore ctx;
	*value = (uint64_t)gum_cpu_context_get_nth_argument(
		gum_ctx->cpu_context, n);
	return 0;
}

extern "C" uint64_t bpftime_get_retval(uint64_t, uint64_t, uint64_t, uint64_t,
				       uint64_t)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EOPNOTSUPP;
	}
	return (uintptr_t)gum_invocation_context_get_return_value(gum_ctx);
}
