#ifndef _NV_ATTACH_FATBIN_RECORD
#define _NV_ATTACH_FATBIN_RECORD

#include "cuda.h"
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
namespace bpftime
{
namespace attach
{
struct ptx_in_module {
	CUmodule module_ptr;
	CUcontext context;
	ptx_in_module(CUmodule module_ptr, CUcontext context)
		: module_ptr(module_ptr), context(context)
	{
	}
	virtual ~ptx_in_module();
};
struct variable_info {
	std::string symbol_name;
	CUdeviceptr ptr;
	size_t size;
	ptx_in_module *ptx;
};

struct kernel_info {
	std::string symbol_name;
	CUfunction func;
	ptx_in_module *ptx;
};
struct fatbin_record {
	using module_key = std::pair<CUcontext, std::string>;
	std::shared_ptr<std::map<module_key, std::shared_ptr<ptx_in_module>>>
		module_pool;
	std::shared_ptr<std::map<std::string, std::vector<uint8_t>>> ptx_pool;
	std::map<void *, std::string> variable_addr_to_symbol;
	std::map<void *, std::string> function_addr_to_symbol;
	std::map<std::string, std::string> original_ptx;
	bool all_ptx_not_modified = true;
	void try_loading_ptxs(class nv_attach_impl &);
	virtual ~fatbin_record();
	bool find_and_fill_variable_info(void *ptr, const char *symbol_name);
	bool find_and_fill_function_info(void *ptr, const char *symbol_name);
	std::optional<variable_info>
	find_variable_info(class nv_attach_impl &, void *ptr);
	std::optional<variable_info>
	find_variable_info(class nv_attach_impl &, const std::string &name);
	std::optional<kernel_info>
	find_function_info(class nv_attach_impl &, void *ptr);
	std::optional<kernel_info>
	find_function_info(class nv_attach_impl &, const std::string &name);

    private:
	std::map<CUcontext, std::vector<std::shared_ptr<ptx_in_module>>>
		ptxs_by_context;
	std::mutex load_mutex;
	std::map<std::string, std::vector<uint8_t>>
	compile_ptxs(class nv_attach_impl &impl,std::map<std::string, std::tuple<std::string, bool>>);
};

} // namespace attach
} // namespace bpftime

#endif
