#include "nv_attach_fatbin_record.hpp"
#include "cuda.h"
#include "nvPTXCompiler.h"
#include "nv_attach_utils.hpp"
#include "spdlog/spdlog.h"
#include "nv_attach_impl.hpp"
#include <algorithm>
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <dlfcn.h>
#include <future>
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
	CUcontext current = nullptr;
	const bool switch_context =
		context != nullptr &&
		(cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
		 current != context);
	if (switch_context && cuCtxPushCurrent(context) != CUDA_SUCCESS) {
		SPDLOG_ERROR(
			"Unable to switch context before unloading module");
		return;
	}
	CUDA_DRIVER_CHECK_NO_EXCEPTION(cuModuleUnload(this->module_ptr),
				       "Unable to unload module");
	if (switch_context) {
		CUcontext popped = nullptr;
		CUDA_DRIVER_CHECK_NO_EXCEPTION(
			cuCtxPopCurrent(&popped),
			"Unable to restore CUDA context");
	}
}

bool fatbin_record::find_and_fill_variable_info(void *ptr,
						const char *symbol_name,
						int device_ordinal)
{
	for (const auto &ptx : get_ptxs_for_device(device_ordinal)) {
		CUdeviceptr dptr;
		size_t size;
		auto err = cuModuleGetGlobal(&dptr, &size, ptx->module_ptr,
					     symbol_name);
		if (err == CUDA_SUCCESS) {
			std::lock_guard<std::mutex> guard(symbol_mutex);
			variable_addr_to_symbol[{ device_ordinal, ptr }] =
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

bool fatbin_record::ensure_variable_info(class nv_attach_impl &impl, void *ptr,
					 int device_ordinal)
{
	std::string symbol_name;
	{
		std::lock_guard<std::mutex> guard(symbol_mutex);
		if (variable_addr_to_symbol.contains({ device_ordinal, ptr }))
			return true;
		auto existing =
			std::find_if(variable_addr_to_symbol.begin(),
				     variable_addr_to_symbol.end(),
				     [ptr](const auto &entry) {
					     return entry.first.second == ptr;
				     });
		if (existing == variable_addr_to_symbol.end())
			return false;
		symbol_name = existing->second.symbol_name;
	}

	std::string sm_arch = impl.get_device_manager().device_count() > 0 ?
				      impl.get_device_manager()
					      .get_device(device_ordinal)
					      .sm_arch :
				      get_gpu_sm_arch();
	try {
		try_loading_ptxs_for_device(impl, device_ordinal, sm_arch);
	} catch (const std::exception &ex) {
		SPDLOG_ERROR("Unable to load PTX for device {}: {}",
			     device_ordinal, ex.what());
		return false;
	}
	return find_and_fill_variable_info(ptr, symbol_name.c_str(),
					   device_ordinal);
}

std::optional<variable_info>
fatbin_record::find_variable_info(void *ptr, int device_ordinal) const
{
	std::lock_guard<std::mutex> guard(symbol_mutex);
	auto itr = variable_addr_to_symbol.find({ device_ordinal, ptr });
	if (itr == variable_addr_to_symbol.end())
		return std::nullopt;
	return itr->second;
}

bool fatbin_record::find_and_fill_function_info(void *ptr,
						const char *symbol_name,
						int device_ordinal)
{
	for (const auto &ptx : get_ptxs_for_device(device_ordinal)) {
		CUfunction func;
		auto err = cuModuleGetFunction(&func, ptx->module_ptr,
					       symbol_name);
		if (err == CUDA_SUCCESS) {
			std::lock_guard<std::mutex> guard(symbol_mutex);
			function_addr_to_symbol[{ device_ordinal, ptr }] =
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

std::optional<kernel_info>
fatbin_record::find_function_info(void *ptr,
				  std::optional<int> device_ordinal) const
{
	std::lock_guard<std::mutex> guard(symbol_mutex);
	if (device_ordinal) {
		auto itr =
			function_addr_to_symbol.find({ *device_ordinal, ptr });
		return itr == function_addr_to_symbol.end() ?
			       std::nullopt :
			       std::optional<kernel_info>(itr->second);
	}
	auto itr = std::find_if(
		function_addr_to_symbol.begin(), function_addr_to_symbol.end(),
		[ptr](const auto &entry) { return entry.first.second == ptr; });
	return itr == function_addr_to_symbol.end() ?
		       std::nullopt :
		       std::optional<kernel_info>(itr->second);
}

std::vector<std::shared_ptr<ptx_in_module>>
fatbin_record::get_ptxs_for_device(int device_ordinal) const
{
	std::lock_guard<std::mutex> load_guard(load_mutex);
	std::vector<std::shared_ptr<ptx_in_module>> result;
	for (const auto &ptx : ptxs) {
		if (ptx->device_ordinal == device_ordinal)
			result.push_back(ptx);
	}
	return result;
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
	std::vector<std::future<void>> compile_tasks;
	for (const auto &[name, ptx_and_trampoline_flag] : patched_ptx) {
		const auto &ptx = std::get<0>(ptx_and_trampoline_flag);

		std::packaged_task<void()> compile_task(
			[&handler, ptx, name, &compiled_ptx, &map_lock, this,
			 &impl, sm_arch]() -> void {
				const auto ptx_fixed =
					rewrite_ptx_target(ptx, sm_arch);
				auto sha256_string = sha256(ptx_fixed.data(),
							    ptx_fixed.size());
				std::vector<uint8_t> compiled_program;
				{
					std::lock_guard<std::mutex> pool_guard(
						impl.get_ptx_pool_mutex());
					if (auto itr = this->ptx_pool->find(
						    sha256_string);
					    itr != this->ptx_pool->end()) {
						compiled_program = itr->second;
					}
				}
				if (!compiled_program.empty()) {
					SPDLOG_INFO(
						"PTX {} ({}) found in cache",
						name, sha256_string);
					std::lock_guard<std::mutex> _guard(
						map_lock);
					compiled_ptx[name] =
						std::move(compiled_program);
					return;
				}

				SPDLOG_INFO(
					"Start compiling {}, not found in cache",
					name);
				auto compiler = handler.create();
				if (!compiler) {
					throw std::runtime_error(
						"Unable to create nv_attach_impl_ptx_compiler");
				}
				std::string gpu_name = "--gpu-name=" + sm_arch;
				const char *compile_options[] = {
					gpu_name.c_str(), "--verbose", "-O3"
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
					handler.destroy(compiler);
					throw std::runtime_error(
						"Unable to compile");
				}
				SPDLOG_DEBUG("Info: {}",
					     handler.get_info_log(compiler));
				uint8_t *data;
				size_t size;
				handler.get_compiled_program(compiler, &data,
							     &size);
				compiled_program.assign(data, data + size);
				handler.destroy(compiler);
				bool cache_hit_during_compile = false;
				{
					std::lock_guard<std::mutex> pool_guard(
						impl.get_ptx_pool_mutex());
					if (auto itr = this->ptx_pool->find(
						    sha256_string);
					    itr != this->ptx_pool->end()) {
						cache_hit_during_compile = true;
						compiled_program = itr->second;
					} else {
						this->ptx_pool->emplace(
							sha256_string,
							compiled_program);
					}
				}
				{
					std::lock_guard<std::mutex> _guard(
						map_lock);
					compiled_ptx[name] = compiled_program;
				}
				if (cache_hit_during_compile) {
					SPDLOG_INFO(
						"PTX {} ({}) was cached while compiling",
						name, sha256_string);
				} else {
					SPDLOG_INFO("Compile of {} done", name);
				}
			});
		compile_tasks.push_back(compile_task.get_future());
		boost::asio::post(pool, std::move(compile_task));
	}
	pool.join();
	for (auto &task : compile_tasks)
		task.get();
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
	std::lock_guard<std::mutex> load_guard(load_mutex);
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
		std::shared_ptr<ptx_in_module> cached_module;
		{
			std::lock_guard<std::mutex> module_pool_guard(
				impl.get_module_pool_mutex());
			if (auto itr =
				    effective_module_pool->find(sha256_string);
			    itr != effective_module_pool->end()) {
				cached_module = itr->second;
			}
		}
		if (cached_module) {
			SPDLOG_INFO("Module {} found in cache (device {})",
				    name, device_ordinal);
			ptxs.push_back(cached_module);
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
					(uintptr_t)impl.shared_mem_ptr,
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
						     &impl.shared_mem_ptr,
						     const_data_size),
					"Unable to copy constData pointer to device");
				// Set per-module device ordinal
				CUdeviceptr dev_ord_ptr;
				size_t dev_ord_size;
				if (cuModuleGetGlobal(
					    &dev_ord_ptr, &dev_ord_size, module,
					    "deviceOrdinal") == CUDA_SUCCESS) {
					uint32_t ord = (uint32_t)device_ordinal;
					CUDA_DRIVER_CHECK_EXCEPTION(
						cuMemcpyHtoD(dev_ord_ptr, &ord,
							     sizeof(ord)),
						"Unable to copy deviceOrdinal to device");
					SPDLOG_INFO(
						"deviceOrdinal={} set for module on device {}",
						ord, device_ordinal);
				}
				CUDA_DRIVER_CHECK_EXCEPTION(
					cuMemcpyHtoD(map_basic_info_ptr,
						     impl.map_basic_info->data(),
						     map_basic_info_size),
					"Unable to copy map_info to device");
				SPDLOG_INFO(
					"Trampoline data copied for device {}",
					device_ordinal);
			}
			CUcontext module_context = nullptr;
			CUDA_DRIVER_CHECK_EXCEPTION(
				cuCtxGetCurrent(&module_context),
				"Unable to get module CUDA context");
			auto ptr = std::make_shared<ptx_in_module>(
				module, module_context, device_ordinal);
			{
				std::lock_guard<std::mutex> module_pool_guard(
					impl.get_module_pool_mutex());
				if (auto itr = effective_module_pool->find(
					    sha256_string);
				    itr != effective_module_pool->end()) {
					cached_module = itr->second;
				} else {
					effective_module_pool->emplace(
						sha256_string, ptr);
				}
			}
			if (cached_module) {
				ptxs.push_back(cached_module);
				SPDLOG_INFO(
					"Module {} was cached while loading (device {})",
					name, device_ordinal);
			} else {
				ptxs.push_back(ptr);
				SPDLOG_INFO("Loaded module: {} on device {}",
					    name, device_ordinal);
			}
		}
	}
	devices_loaded.insert(device_ordinal);
}

} // namespace bpftime::attach
