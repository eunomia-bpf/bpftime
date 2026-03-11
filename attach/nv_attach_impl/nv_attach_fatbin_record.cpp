#include "nv_attach_fatbin_record.hpp"
#include "cuda.h"
#include "nvPTXCompiler.h"
#include "nv_attach_utils.hpp"
#include "spdlog/spdlog.h"
#include "nv_attach_impl.hpp"
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <dlfcn.h>
#include <iterator>
#include <stdexcept>
#include <ptx_pass_config.h>
#include "ptx_compiler/ptx_compiler.hpp"
#include <utility>
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
#define NVPTXCOMPILER_CHECK_EXCEPTION(x, message)                              \
	do {                                                                   \
		nvPTXCompileResult result = x;                                 \
		if (result != NVPTXCOMPILE_SUCCESS) {                          \
			SPDLOG_ERROR("error: {} failed with error code {}\n",  \
				     #x, (int)result);                         \
			throw std::runtime_error(message);                     \
		}                                                              \
	} while (0)
namespace bpftime::attach
{
using bpftime::attach::rewrite_ptx_target;

fatbin_record::~fatbin_record()
{
}
ptx_in_module::~ptx_in_module()
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

std::map<std::string, std::vector<uint8_t>> fatbin_record::compile_ptxs(
	class nv_attach_impl &impl,
	std::map<std::string, std::tuple<std::string, bool>> patched_ptx,
	const std::string &sm_arch)
{
	SPDLOG_INFO("Compiling PTXs with sm_arch {}", sm_arch);

	unsigned major, minor;
	NVPTXCOMPILER_CHECK_EXCEPTION(nvPTXCompilerGetVersion(&major, &minor),
				      "Unable to get compiler version");
	SPDLOG_INFO("Compiler version: {}.{}", major, minor);

	std::map<std::string, std::vector<uint8_t>> compiled_ptx;
	const auto &handler = impl.ptx_compiler;
	boost::asio::thread_pool pool(std::thread::hardware_concurrency());
	std::mutex map_lock;
	for (const auto &[name, ptx_and_trampoline_flag] : patched_ptx) {
		const auto &ptx = std::get<0>(ptx_and_trampoline_flag);

			boost::asio::post(
				pool,
				[&handler, ptx, name, &compiled_ptx, &map_lock, this,
				 sm_arch]() -> void {
					const auto ptx_fixed =
						rewrite_ptx_target(ptx, sm_arch);
					auto sha256_string =
						sha256(ptx_fixed.data(), ptx_fixed.size());
					if (auto itr =
						    this->ptx_pool->find(sha256_string);
					    itr != this->ptx_pool->end()) {
						SPDLOG_INFO(
						"PTX {} ({}) found in cache",
						name, sha256_string);
					std::lock_guard<std::mutex> _guard(
						map_lock);
					compiled_ptx[name] = itr->second;
				} else {
					SPDLOG_INFO(
						"Start compiling {}, not found in cache",
						name);
					auto compiler = handler.create();
					if (!compiler) {
						throw std::runtime_error(
							"Unable to create nv_attach_impl_ptx_compiler");
					}
						std::string gpu_name =
							"--gpu-name=" + sm_arch;
							const char *compile_options[] = {
								gpu_name.c_str(), "--verbose",
								"-O3"
							};
						if (auto err = handler.compile(
							    compiler, ptx_fixed.c_str(),
							    compile_options,
							    std::size(compile_options));
						    err != 0) {
							SPDLOG_ERROR(
								"Unable to compile: {}, error = {}",
							err,
							handler.get_error_log(
								compiler));
						throw std::runtime_error(
							"Unable to compile");
					}
					SPDLOG_DEBUG(
						"Info: {}",
						handler.get_info_log(compiler));
					uint8_t *data;
					size_t size;
					handler.get_compiled_program(
						compiler, &data, &size);
					std::vector<uint8_t> compiled_program(
						data, data + size);
					handler.destroy(compiler);
					std::lock_guard<std::mutex> _guard(
						map_lock);
					compiled_ptx[name] = compiled_program;
					this->ptx_pool->insert(std::make_pair(
						sha256_string,
						compiled_program));
					SPDLOG_INFO("Compile of {} done", name);
				}
			});
	}
	pool.join();
	return compiled_ptx;
}
void fatbin_record::try_loading_ptxs(class nv_attach_impl &impl)
{
	// Determine current device ordinal
	int dev_ordinal = impl.get_current_device_ordinal();
	std::string sm_arch;
	if (impl.get_device_manager().device_count() > 0) {
		sm_arch = impl.get_device_manager()
				  .get_device(dev_ordinal)
				  .sm_arch;
	} else {
		sm_arch = get_gpu_sm_arch();
	}
	try_loading_ptxs_for_device(impl, dev_ordinal, sm_arch);
}

void fatbin_record::try_loading_ptxs_for_device(class nv_attach_impl &impl,
						 int device_ordinal,
						 const std::string &sm_arch)
{
	// Skip if already loaded for this device
	if (devices_loaded.count(device_ordinal) > 0)
		return;
	if (impl.shared_mem_ptr == 0) {
		throw std::runtime_error(
			"shared_mem_ptr is not initialized before loading PTX");
	}
	SPDLOG_INFO(
		"Loading & patching current fatbin for device {} (sm_arch {})..",
		device_ordinal, sm_arch);

	auto patched_ptx = *impl.hack_fatbin(original_ptx);

	auto compiled_ptx = compile_ptxs(impl, patched_ptx, sm_arch);

	// Use per-device module pool if available
	auto &dev_manager = impl.get_device_manager();
	auto effective_module_pool =
		(dev_manager.device_count() > device_ordinal) ?
			dev_manager.get_device(device_ordinal).module_pool :
			module_pool;

	for (const auto &[name, ptx_and_trampoline_flag] : patched_ptx) {
		const auto &ptx = std::get<0>(ptx_and_trampoline_flag);
		bool added_trampoline = std::get<1>(ptx_and_trampoline_flag);
		const auto &compiled_elf = compiled_ptx.at(name);
		// Include device ordinal in cache key so different devices get
		// separate modules
		auto sha256_string =
			sha256(compiled_elf.data(), compiled_elf.size()) +
			":dev" + std::to_string(device_ordinal);
		if (auto itr = effective_module_pool->find(sha256_string);
		    itr != effective_module_pool->end()) {
			SPDLOG_INFO("Module {} found in cache (device {})",
				    name, device_ordinal);
			ptxs.push_back(itr->second);
		} else {
			CUmodule module;
			SPDLOG_INFO(
				"Loading module: {} for device {}, not found in cache",
				name, device_ordinal);
			char error_buf[8192] = { 0 }, info_buf[8192] = { 0 };
			CUjit_option options[] = {
				CU_JIT_INFO_LOG_BUFFER,
				CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
				CU_JIT_ERROR_LOG_BUFFER,
				CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
			};
			void *option_values[] = {
				(void *)info_buf, (void *)std::size(info_buf),
				(void *)error_buf, (void *)std::size(error_buf)
			};
			if (auto err = cuModuleLoadDataEx(
				    &module, compiled_elf.data(),
				    std::size(options), options, option_values);
			    err != CUDA_SUCCESS) {
				SPDLOG_ERROR(
					"Unable to compile module {} on device {}: {}",
					name, device_ordinal, (int)err);
				SPDLOG_ERROR("Info: {}", info_buf);
				SPDLOG_ERROR("Error: {}", error_buf);
				throw std::runtime_error(
					"Unable to compile module");
			}
			if (added_trampoline) {
				// Determine the shared_mem_ptr to use for this
				// device
				uintptr_t device_shared_mem_ptr =
					impl.shared_mem_ptr;
				if (dev_manager.device_count() >
				    device_ordinal) {
					auto &dev_info =
						dev_manager.get_device(
							device_ordinal);
					if (dev_info.shared_mem_device_ptr !=
					    0) {
						device_shared_mem_ptr =
							dev_info.shared_mem_device_ptr;
					}
				}

				CUdeviceptr const_data_ptr, map_basic_info_ptr;
				size_t const_data_size, map_basic_info_size;
				SPDLOG_INFO(
					"Copying trampoline data to device {}",
					device_ordinal);
				CUDA_DRIVER_CHECK_EXCEPTION(
					cuModuleGetGlobal(&const_data_ptr,
							  &const_data_size,
							  module, "constData"),
					"Unable to get pointer of constData");
				SPDLOG_INFO(
					"constData symbol device_ptr={:x} size={} shared_mem_ptr={:x} (device {})",
					(uintptr_t)const_data_ptr,
					const_data_size,
					(uintptr_t)device_shared_mem_ptr,
					device_ordinal);
				CUDA_DRIVER_CHECK_EXCEPTION(
					cuModuleGetGlobal(&map_basic_info_ptr,
							  &map_basic_info_size,
							  module, "map_info"),
					"Unable to get pointer of map_info");
				SPDLOG_INFO(
					"map_info symbol device_ptr={:x} size={}",
					(uintptr_t)map_basic_info_ptr,
					map_basic_info_size);
				CUDA_DRIVER_CHECK_EXCEPTION(
					cuMemcpyHtoD(const_data_ptr,
						     &device_shared_mem_ptr,
						     const_data_size),
					"Unable to copy constData pointer to device");
				// Set per-module device ordinal
				CUdeviceptr dev_ord_ptr;
				size_t dev_ord_size;
				if (cuModuleGetGlobal(&dev_ord_ptr,
						      &dev_ord_size, module,
						      "deviceOrdinal") ==
				    CUDA_SUCCESS) {
					uint32_t ord =
						(uint32_t)device_ordinal;
					CUDA_DRIVER_CHECK_EXCEPTION(
						cuMemcpyHtoD(dev_ord_ptr, &ord,
							     sizeof(ord)),
						"Unable to copy deviceOrdinal to device");
					SPDLOG_INFO(
						"deviceOrdinal={} set for module on device {}",
						ord, device_ordinal);
				}
				CUDA_DRIVER_CHECK_EXCEPTION(
					cuMemcpyHtoD(
						map_basic_info_ptr,
						impl.map_basic_info->data(),
						map_basic_info_size),
					"Unable to copy map_info to device");
				SPDLOG_INFO(
					"Trampoline data copied for device {}",
					device_ordinal);
			}
			auto ptr = std::make_shared<ptx_in_module>(
				module, device_ordinal);
			effective_module_pool->insert(
				std::make_pair(sha256_string, ptr));
			ptxs.push_back(ptr);
			SPDLOG_INFO("Loaded module: {} on device {}", name,
				    device_ordinal);
		}
	}
	devices_loaded.insert(device_ordinal);
	ptx_loaded = true;
}

} // namespace bpftime::attach
