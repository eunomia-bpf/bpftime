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
// transform from callback variant index to attach type
int from_cb_idx_to_attach_type(int idx);

// Helper implementation of bpf_get_func_ret
extern "C" uint64_t bpftime_get_func_ret(uint64_t ctx, uint64_t *value,
					 uint64_t, uint64_t, uint64_t);
// Helper implementation of bpf_get_func_arg
extern "C" uint64_t bpftime_get_func_arg(uint64_t ctx, uint32_t n,
					 uint64_t *value, uint64_t, uint64_t);
// Helper implementation of bpf_get_retval
extern "C" uint64_t bpftime_get_retval(uint64_t, uint64_t, uint64_t, uint64_t,
				       uint64_t);
} // namespace attach
} // namespace bpftime
#endif
