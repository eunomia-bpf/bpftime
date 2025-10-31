#include "nv_attach_fatbin_record.hpp"
#include "cuda.h"
#include "spdlog/spdlog.h"
#include "nv_attach_impl.hpp"
#define CUDA_DRIVER_CHECK_NO_EXCEPTION(expr, message)                          \
	do {                                                                   \
		if (auto err = expr; err != CUDA_SUCCESS) {                    \
			SPDLOG_ERROR("{}: {}", message, (int)err);             \
		}                                                              \
	} while (false)
#define CUDA_DRIVER_CHECK_EXCEPTION(expr, message)                             \
	do {                                                                   \
		if (auto err = expr; err != CUDA_SUCCESS) {                    \
			SPDLOG_ERROR("{}: {}", message, (int)err);             \
			throw std::runtime_error(message);                     \
		}                                                              \
	} while (false)

namespace bpftime::attach
{
fatbin_record::~fatbin_record()
{
}
fatbin_record::ptx_in_module::~ptx_in_module()
{
	CUDA_DRIVER_CHECK_NO_EXCEPTION(cuModuleUnload(this->module_ptr),
				       "Unable to unload module");
}

bool fatbin_record::find_and_fill_variable_info(void *ptr,
						const char *symbol_name)
{
	for (const auto &ptx : ptxs) {
		CUdeviceptr dptr;
		size_t size;
		auto err = cuModuleGetGlobal(&dptr, &size, ptx->module_ptr,
					     symbol_name);
		if (err == CUDA_SUCCESS) {
			variable_addr_to_symbol[ptr] =
				variable_info{ .symbol_name =
						       std::string(symbol_name),
					       .ptr = dptr,
					       .size = size,
					       .ptx = ptx.get() };
			return true;
		} else if (err == CUDA_ERROR_NOT_FOUND) {
			continue;
		} else {
			SPDLOG_ERROR("Unable to lookup symbol: {}", (int)err);
			return false;
		}
	}
	return false;
}
bool fatbin_record::find_and_fill_function_info(void *ptr,
						const char *symbol_name)
{
	for (const auto &ptx : ptxs) {
		CUfunction func;
		auto err = cuModuleGetFunction(&func, ptx->module_ptr,
					       symbol_name);
		if (err == CUDA_SUCCESS) {
			function_addr_to_symbol[ptr] =
				kernel_info{ .symbol_name =
						     std::string(symbol_name),
					     .func = func,
					     .ptx = ptx.get() };
			return true;
		} else if (err == CUDA_ERROR_NOT_FOUND) {
			continue;
		} else {
			SPDLOG_ERROR("Unable to lookup function: {}", (int)err);
			return false;
		}
	}
	return false;
}

void fatbin_record::try_loading_ptxs(class nv_attach_impl &impl)
{
	if (ptx_loaded)
		return;
	SPDLOG_INFO("Loading & patching current fatbin..");
	auto patched_ptx = *impl.hack_fatbin(original_ptx);

	for (const auto &[name, ptx] : patched_ptx) {
		CUmodule module;
		SPDLOG_INFO("Loading module: {}", name);
		char error_buf[8192], info_buf[8192];
		CUjit_option options[] = { CU_JIT_INFO_LOG_BUFFER,
					   CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
					   CU_JIT_ERROR_LOG_BUFFER,
					   CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
		void *option_values[] = { (void *)info_buf,
					  (void *)std::size(info_buf),
					  (void *)error_buf,
					  (void *)std::size(error_buf) };
		if (auto err = cuModuleLoadDataEx(&module, ptx.data(),
						  std::size(options), options,
						  option_values);
		    err != CUDA_SUCCESS) {
			SPDLOG_ERROR("Unable to compile module {}: {}", name,
				     (int)err);
			SPDLOG_ERROR("Info: {}", info_buf);
			SPDLOG_ERROR("Error: {}", error_buf);
			throw std::runtime_error("Unable to compile module");
		}
		CUdeviceptr const_data_ptr, map_basic_info_ptr;
		size_t const_data_size, map_basic_info_size;
		SPDLOG_INFO("Copying trampoline data to device");
		CUDA_DRIVER_CHECK_EXCEPTION(
			cuModuleGetGlobal(&const_data_ptr, &const_data_size,
					  module, "constData"),
			"Unable to get pointer of constData");
		CUDA_DRIVER_CHECK_EXCEPTION(
			cuModuleGetGlobal(&map_basic_info_ptr,
					  &map_basic_info_size, module,
					  "map_info"),
			"Unable to get pointer of map_info");
		CUDA_DRIVER_CHECK_EXCEPTION(
			cuMemcpyHtoD(const_data_ptr, &impl.shared_mem_ptr,
				     const_data_size),
			"Unable to copy constData pointer to device");
		CUDA_DRIVER_CHECK_EXCEPTION(
			cuMemcpyHtoD(map_basic_info_ptr,
				     impl.map_basic_info->data(),
				     map_basic_info_size),
			"Unable to copy constData pointer to device");
		SPDLOG_INFO("Trampoline data copied");
		ptxs.emplace_back(
			std::make_unique<fatbin_record::ptx_in_module>(module));
		SPDLOG_INFO("Loaded module: {}", name);
	}
	ptx_loaded = true;
}

} // namespace bpftime::attach
