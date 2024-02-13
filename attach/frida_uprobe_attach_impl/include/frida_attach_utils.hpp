#ifndef _BPFTIME_FRIDA_ATTACH_UTILS_HPP
#define _BPFTIME_FRIDA_ATTACH_UTILS_HPP
#include <string_view>
#include <cstdint>
namespace bpftime
{
namespace attach
{
// Return the function address in a certain module, providing the function
// offset in the module
void *
resolve_function_addr_by_module_offset(const std::string_view &module_name,
				       uintptr_t func_offset);
// Lookup a function address by its symbol name
void *find_function_addr_by_name(const char *name);
// Get the base address of a certain module
void *get_module_base_addr(const char *module_name);
// Find the symbol address of an exported symbol of a certain module
void *find_module_export_by_name(const char *module_name,
				 const char *symbol_name);
} // namespace attach
} // namespace bpftime
#endif
