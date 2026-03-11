#ifndef _NV_ATTACH_FATBIN_RECORD
#define _NV_ATTACH_FATBIN_RECORD

#include "cuda.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
namespace bpftime
{
namespace attach
{
struct ptx_in_module {
	CUmodule module_ptr;
	int device_ordinal; // which device this module was loaded on
	ptx_in_module(CUmodule module_ptr, int device_ordinal = 0)
		: module_ptr(module_ptr), device_ordinal(device_ordinal)
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
	/// Track which devices have had their modules loaded
	std::set<int> devices_loaded;

	/// Load PTXs for the current CUDA context's device (backward compat)
	void try_loading_ptxs(class nv_attach_impl &);
	/// Load PTXs for a specific device
	void try_loading_ptxs_for_device(class nv_attach_impl &impl,
					  int device_ordinal,
					  const std::string &sm_arch);
	virtual ~fatbin_record();
	bool find_and_fill_variable_info(void *ptr, const char *symbol_name);
	bool find_and_fill_function_info(void *ptr, const char *symbol_name);

    private:
	std::map<std::string, std::vector<uint8_t>>
	compile_ptxs(class nv_attach_impl &impl,
		     std::map<std::string, std::tuple<std::string, bool>>,
		     const std::string &sm_arch);
};

} // namespace attach
} // namespace bpftime

#endif
