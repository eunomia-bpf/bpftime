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
#include <algorithm>
#include <exception>
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
		(cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != context);
	bool pushed = false;
	if (switch_context) {
		if (cuCtxPushCurrent(context) == CUDA_SUCCESS)
			pushed = true;
		else
			SPDLOG_ERROR(
				"Unable to switch context before unloading module");
	}
	CUDA_DRIVER_CHECK_NO_EXCEPTION(cuModuleUnload(this->module_ptr),
				       "Unable to unload module");
	if (pushed) {
		CUcontext popped = nullptr;
		CUDA_DRIVER_CHECK_NO_EXCEPTION(
			cuCtxPopCurrent(&popped),
			"Unable to restore CUDA context after unloading module");
	}
}

bool fatbin_record::find_and_fill_variable_info(void *ptr,
						const char *symbol_name)
{
	if (ptr == nullptr || symbol_name == nullptr)
		return false;
	variable_addr_to_symbol[ptr] = symbol_name;
	return true;
}
bool fatbin_record::find_and_fill_function_info(void *ptr,
						const char *symbol_name)
{
	if (ptr == nullptr || symbol_name == nullptr)
		return false;
	function_addr_to_symbol[ptr] = symbol_name;
	return true;
}

std::optional<variable_info>
fatbin_record::find_variable_info(nv_attach_impl &impl, void *ptr)
{
	auto itr = variable_addr_to_symbol.find(ptr);
	if (itr == variable_addr_to_symbol.end())
		return std::nullopt;
	return find_variable_info(impl, itr->second);
}

std::optional<variable_info>
fatbin_record::find_variable_info(nv_attach_impl &impl,
				  const std::string &name)
{
	try {
		try_loading_ptxs(impl);
	} catch (const std::exception &ex) {
		SPDLOG_ERROR("Unable to load PTX while resolving {}: {}", name,
			     ex.what());
		return std::nullopt;
	}
	CUcontext context = nullptr;
	if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr)
		return std::nullopt;
	std::lock_guard<std::mutex> guard(load_mutex);
	for (const auto &ptx : ptxs_by_context.at(context)) {
		CUdeviceptr ptr = 0;
		size_t size = 0;
		auto err = cuModuleGetGlobal(&ptr, &size, ptx->module_ptr,
					     name.c_str());
		if (err == CUDA_SUCCESS)
			return variable_info{ name, ptr, size, ptx.get() };
		if (err != CUDA_ERROR_NOT_FOUND) {
			SPDLOG_ERROR("Unable to lookup symbol {}: {}", name,
				     (int)err);
			return std::nullopt;
		}
	}
	return std::nullopt;
}

std::optional<kernel_info>
fatbin_record::find_function_info(nv_attach_impl &impl, void *ptr)
{
	auto itr = function_addr_to_symbol.find(ptr);
	if (itr == function_addr_to_symbol.end())
		return std::nullopt;
	return find_function_info(impl, itr->second);
}

std::optional<kernel_info>
fatbin_record::find_function_info(nv_attach_impl &impl,
				  const std::string &name)
{
	try {
		try_loading_ptxs(impl);
	} catch (const std::exception &ex) {
		SPDLOG_ERROR("Unable to load PTX while resolving {}: {}", name,
			     ex.what());
		return std::nullopt;
	}
	CUcontext context = nullptr;
	if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr)
		return std::nullopt;
	std::lock_guard<std::mutex> guard(load_mutex);
	for (const auto &ptx : ptxs_by_context.at(context)) {
		CUfunction function = nullptr;
		auto err = cuModuleGetFunction(&function, ptx->module_ptr,
					       name.c_str());
		if (err == CUDA_SUCCESS)
			return kernel_info{ name, function, ptx.get() };
		if (err != CUDA_ERROR_NOT_FOUND) {
			SPDLOG_ERROR("Unable to lookup function {}: {}", name,
				     (int)err);
			return std::nullopt;
		}
	}
	return std::nullopt;
}

std::map<std::string, std::vector<uint8_t>> fatbin_record::compile_ptxs(
	class nv_attach_impl &impl,
	std::map<std::string, std::tuple<std::string, bool>> patched_ptx)
{
	std::string sm_arch = get_gpu_sm_arch();
	SPDLOG_INFO("Compiling PTXs with sm_arch {}", sm_arch);

	unsigned major, minor;
	NVPTXCOMPILER_CHECK_EXCEPTION(nvPTXCompilerGetVersion(&major, &minor),
				      "Unable to get compiler version");
	SPDLOG_INFO("Compiler version: {}.{}", major, minor);

	std::map<std::string, std::vector<uint8_t>> compiled_ptx;
	const auto &handler = impl.ptx_compiler;
	boost::asio::thread_pool pool(
		std::max(1u, std::thread::hardware_concurrency()));
	std::mutex map_lock;
	std::exception_ptr error;
	for (const auto &[name, ptx_and_trampoline_flag] : patched_ptx) {
		const auto &ptx = std::get<0>(ptx_and_trampoline_flag);

		boost::asio::post(pool, [&, ptx, name, sm_arch]() {
			try {
				const auto ptx_fixed =
					rewrite_ptx_target(ptx, sm_arch);
				const auto sha256_string = sha256(
					ptx_fixed.data(), ptx_fixed.size());
				std::vector<uint8_t> compiled_program;
				{
					std::lock_guard<std::mutex> guard(
						impl.ptx_cache_mutex);
					if (auto itr = ptx_pool->find(sha256_string);
					    itr != ptx_pool->end())
						compiled_program = itr->second;
				}
				if (compiled_program.empty()) {
					SPDLOG_INFO(
						"Start compiling {}, not found in cache",
						name);
					std::unique_ptr<nv_attach_impl_ptx_compiler,
							decltype(handler.destroy)>
						compiler(handler.create(),
							 handler.destroy);
					if (!compiler)
						throw std::runtime_error(
							"Unable to create nv_attach_impl_ptx_compiler");
					std::string gpu_name = "--gpu-name=" + sm_arch;
					const char *compile_options[] = {
						gpu_name.c_str(), "--verbose", "-O3"
					};
					if (auto err = handler.compile(
						    compiler.get(), ptx_fixed.c_str(),
						    compile_options,
						    std::size(compile_options));
					    err != 0) {
						SPDLOG_ERROR(
							"Unable to compile: {}, error = {}",
							err, handler.get_error_log(
								     compiler.get()));
						throw std::runtime_error(
							"Unable to compile");
					}
					uint8_t *data;
					size_t size;
					handler.get_compiled_program(
						compiler.get(), &data, &size);
					compiled_program.assign(data, data + size);
					std::lock_guard<std::mutex> guard(
						impl.ptx_cache_mutex);
					const auto [itr, inserted] = ptx_pool->emplace(
						sha256_string, compiled_program);
					if (!inserted)
						compiled_program = itr->second;
				} else {
					SPDLOG_INFO("PTX {} ({}) found in cache", name,
						    sha256_string);
				}
				std::lock_guard<std::mutex> guard(map_lock);
				compiled_ptx[name] = std::move(compiled_program);
			} catch (...) {
				std::lock_guard<std::mutex> guard(map_lock);
				if (!error)
					error = std::current_exception();
			}
		});
	}
	pool.join();
	if (error)
		std::rethrow_exception(error);
	return compiled_ptx;
}
void fatbin_record::try_loading_ptxs(class nv_attach_impl &impl)
{
	CUcontext context = nullptr;
	if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr)
		throw std::runtime_error("No current CUDA context");
	std::lock_guard<std::mutex> load_guard(load_mutex);
	if (ptxs_by_context.contains(context))
		return;
	if (impl.shared_mem_ptr == 0) {
		throw std::runtime_error(
			"shared_mem_ptr is not initialized before loading PTX");
	}
	SPDLOG_INFO("Loading & patching current fatbin..");

	auto patched_ptx_result = impl.hack_fatbin(original_ptx);
	if (!patched_ptx_result)
		throw std::runtime_error("Unable to patch fatbin PTX");
	auto &patched_ptx = *patched_ptx_result;

	auto compiled_ptx = compile_ptxs(impl, patched_ptx);

	std::vector<std::shared_ptr<ptx_in_module>> context_ptxs;
	for (const auto &[name, ptx_and_trampoline_flag] : patched_ptx) {
		bool added_trampoline = std::get<1>(ptx_and_trampoline_flag);
		const auto &compiled_elf = compiled_ptx.at(name);
		module_key key{ context, sha256(compiled_elf.data(),
					       compiled_elf.size()) };
		std::shared_ptr<ptx_in_module> cached;
		{
			std::lock_guard<std::mutex> guard(impl.module_cache_mutex);
			if (auto itr = module_pool->find(key);
			    itr != module_pool->end())
				cached = itr->second;
		}
		if (cached) {
			SPDLOG_INFO("Module {} found in cache", name);
			context_ptxs.push_back(std::move(cached));
		} else {
			CUmodule module;
			SPDLOG_INFO("Loading module: {}, not found in cache",
				    name);
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
				SPDLOG_ERROR("Unable to compile module {}: {}",
					     name, (int)err);
				SPDLOG_ERROR("Info: {}", info_buf);
				SPDLOG_ERROR("Error: {}", error_buf);
				throw std::runtime_error(
					"Unable to compile module");
			}
			auto ptr =
				std::make_shared<ptx_in_module>(module, context);
			if (added_trampoline) {
				CUdeviceptr const_data_ptr, map_basic_info_ptr;
				size_t const_data_size, map_basic_info_size;
				SPDLOG_INFO(
					"Copying trampoline data to device");
				CUDA_DRIVER_CHECK_EXCEPTION(
					cuModuleGetGlobal(&const_data_ptr,
							  &const_data_size,
							  module, "constData"),
					"Unable to get pointer of constData");
				SPDLOG_INFO(
			"constData symbol device_ptr={:x} size={} shared_mem_ptr={:x}",
			(uintptr_t)const_data_ptr, const_data_size,
			(uintptr_t)impl.shared_mem_ptr);
		CUDA_DRIVER_CHECK_EXCEPTION(
					cuModuleGetGlobal(&map_basic_info_ptr,
							  &map_basic_info_size,
							  module, "map_info"),
					"Unable to get pointer of map_info");
				SPDLOG_INFO("map_info symbol device_ptr={:x} size={}",
			    (uintptr_t)map_basic_info_ptr, map_basic_info_size);
		CUDA_DRIVER_CHECK_EXCEPTION(
					cuMemcpyHtoD(const_data_ptr,
						     &impl.shared_mem_ptr,
						     const_data_size),
					"Unable to copy constData pointer to device");
				CUDA_DRIVER_CHECK_EXCEPTION(
					cuMemcpyHtoD(map_basic_info_ptr,
						     impl.map_basic_info->data(),
						     map_basic_info_size),
					"Unable to copy constData pointer to device");
				SPDLOG_INFO("Trampoline data copied");
			}
			{
				std::lock_guard<std::mutex> guard(
					impl.module_cache_mutex);
				const auto [itr, inserted] =
					module_pool->emplace(key, ptr);
				if (!inserted)
					ptr = itr->second;
			}
			context_ptxs.push_back(std::move(ptr));
			SPDLOG_INFO("Loaded module: {}", name);
		}
	}
	ptxs_by_context.emplace(context, std::move(context_ptxs));
}

} // namespace bpftime::attach
