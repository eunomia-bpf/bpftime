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
};
struct fatbin_record {
	std::shared_ptr<std::map<std::string, std::vector<uint8_t>>> ptx_pool;
	// Zero for records observed through CUDA registration hooks. Non-zero
	// late-bootstrap generations let reruns supersede, but safely retain,
	// modules that may still have in-flight launches.
	std::size_t late_bootstrap_generation = 0;
	std::map<void *, std::string> variable_addr_to_symbol;
	std::map<void *, std::string> function_addr_to_symbol;
	std::map<std::string, std::string> original_ptx;
	bool all_ptx_not_modified = true;
	void try_loading_ptxs(class nv_attach_impl &);
	virtual ~fatbin_record();
	std::optional<variable_info>
	find_variable_info(class nv_attach_impl &, void *ptr);
	std::optional<variable_info>
	find_variable_info(class nv_attach_impl &, const std::string &name);
	std::optional<CUfunction> find_function_info(class nv_attach_impl &,
						     void *ptr);
	std::optional<CUfunction> find_function_info(class nv_attach_impl &,
						     const std::string &name);

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
