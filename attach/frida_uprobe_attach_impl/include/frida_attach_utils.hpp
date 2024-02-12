#ifndef _BPFTIME_FRIDA_ATTACH_UTILS_HPP
#define _BPFTIME_FRIDA_ATTACH_UTILS_HPP
#include <string_view>
#include <cstdint>
namespace bpftime
{
namespace attach
{
void *
resolve_function_addr_by_module_offset(const std::string_view &module_name,
				       uintptr_t func_offset);
void *find_function_addr_by_name(const char *name);
void *get_module_base_addr(const char *module_name);
void *find_module_export_by_name(const char *module_name,
				 const char *symbol_name);
} // namespace attach
} // namespace bpftime
#endif
