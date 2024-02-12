#include "frida_attach_utils.hpp"
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
	if (std::filesystem::equivalent(module_name, exec_path)) {
		module_base_addr = get_module_base_addr("");
	} else {
		module_base_addr =
			get_module_base_addr(std::string(module_name).c_str());
	}
	if (!module_base_addr) {
		SPDLOG_ERROR("Failed to find module base address for {}",
			     module_name);
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

} // namespace attach
} // namespace bpftime
