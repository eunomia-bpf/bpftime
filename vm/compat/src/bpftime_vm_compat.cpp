#include "spdlog/spdlog.h"
#include <bpftime_vm_compat.hpp>

using namespace bpftime::vm::compat;

bpftime_vm_impl::~bpftime_vm_impl()
{
}
bool bpftime_vm_impl::toggle_bounds_check(bool enable)
{
	SPDLOG_CRITICAL("Not implemented yet: toggle_bounds_check");
	return false;
}
void bpftime_vm_impl::register_error_print_callback(int (*fn)(FILE *,
							      const char *,
							      ...))
{
	SPDLOG_CRITICAL("Not implemented yet: register_error_print_callback");
}
int bpftime_vm_impl::register_external_function(size_t index,
						const std::string &name,
						void *fn)
{
	SPDLOG_CRITICAL("Not implemented yet: register_external_function");
	return -1;
}

void bpftime_vm_impl::unload_code()
{
	SPDLOG_CRITICAL("Not implemented yet: unload_code");
}
std::optional<precompiled_ebpf_function> bpftime_vm_impl::compile()
{
	SPDLOG_CRITICAL("Not implemented yet: compile");
	return {};
}

void bpftime_vm_impl::set_lddw_helpers(uint64_t (*map_by_fd)(uint32_t),
				       uint64_t (*map_by_idx)(uint32_t),
				       uint64_t (*map_val)(uint64_t),
				       uint64_t (*var_addr)(uint32_t),
				       uint64_t (*code_addr)(uint32_t))
{
	SPDLOG_CRITICAL("Not implemented yet: set_lddw_helpers");
}
std::string bpftime_vm_impl::get_error_message()
{
	SPDLOG_CRITICAL("Not implemented yet: get_error_message");
	return "";
}
int bpftime_vm_impl::set_pointer_secret(uint64_t secret)
{
	SPDLOG_CRITICAL("Not implemented yet: set_pointer_secret");
	return -1;
}
int bpftime_vm_impl::set_unwind_function_index(size_t idx)
{
	SPDLOG_CRITICAL("Not implemented yet: set_unwind_function_index");
	return -1;
}
