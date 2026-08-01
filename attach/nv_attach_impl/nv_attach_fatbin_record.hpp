#ifndef _NV_ATTACH_FATBIN_RECORD
#define _NV_ATTACH_FATBIN_RECORD

#include "cuda.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace bpftime
{
namespace attach
{
struct ptx_in_module {
	CUmodule module_ptr;
	ptx_in_module(CUmodule module_ptr) : module_ptr(module_ptr)
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
	std::shared_ptr<std::map<std::string, std::shared_ptr<ptx_in_module>>>
		module_pool;
	std::shared_ptr<std::map<std::string, std::vector<uint8_t>>> ptx_pool;
	std::vector<std::shared_ptr<ptx_in_module>> ptxs;
	std::map<void *, variable_info> variable_addr_to_symbol;
	std::map<void *, kernel_info> function_addr_to_symbol;
	std::map<std::string, std::string> original_ptx;
	bool all_ptx_not_modified = true;
	bool ptx_loaded = false;
	void try_loading_ptxs(class nv_attach_impl &);
	virtual ~fatbin_record();
	bool find_and_fill_variable_info(void *ptr, const char *symbol_name);
	bool find_and_fill_function_info(void *ptr, const char *symbol_name);

    private:
	std::map<std::string, std::vector<uint8_t>>
	compile_ptxs(class nv_attach_impl &impl,std::map<std::string, std::tuple<std::string, bool>>);
};

} // namespace attach
} // namespace bpftime

#endif
