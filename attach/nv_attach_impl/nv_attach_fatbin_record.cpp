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
#include <exception>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <thread>
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
	std::mutex error_lock;
	std::exception_ptr first_error;
	for (const auto &[name, ptx_and_trampoline_flag] : patched_ptx) {
		const auto &ptx = std::get<0>(ptx_and_trampoline_flag);

		boost::asio::post(
			pool,
			[&handler, ptx, name, &compiled_ptx, &map_lock,
			 &error_lock, &first_error, this, sm_arch]() -> void {
				try {
					const auto ptx_fixed =
						rewrite_ptx_target(ptx,
								   sm_arch);
					auto sha256_string =
						sha256(ptx_fixed.data(),
						       ptx_fixed.size());
					{
						// ptx_pool and compiled_ptx are
						// shared by every worker in
						// this compilation batch. A
						// lookup must be serialized
						// with inserts as well as other
						// writes.
						std::lock_guard<std::mutex>
							guard(map_lock);
						if (auto itr = this->ptx_pool->find(
							    sha256_string);
						    itr !=
						    this->ptx_pool->end()) {
							SPDLOG_INFO(
								"PTX {} ({}) found in cache",
								name,
								sha256_string);
							compiled_ptx[name] =
								itr->second;
							return;
						}
					}
					{
						SPDLOG_INFO(
							"Start compiling {}, not found in cache",
							name);
						auto compiler = std::unique_ptr<
							nv_attach_impl_ptx_compiler,
							decltype(handler.destroy)>(
							handler.create(),
							handler.destroy);
						if (!compiler) {
							throw std::runtime_error(
								"Unable to create nv_attach_impl_ptx_compiler");
						}
						std::string gpu_name =
							"--gpu-name=" + sm_arch;
						const char *compile_options[] = {
							gpu_name.c_str(),
							"--verbose", "-O3"
						};
						if (auto err = handler.compile(
							    compiler.get(),
							    ptx_fixed.c_str(),
							    compile_options,
							    std::size(
								    compile_options));
						    err != 0) {
							SPDLOG_ERROR(
								"Unable to compile: {}, error = {}",
								err,
								handler.get_error_log(
									compiler.get()));
							throw std::runtime_error(
								"Unable to compile");
						}
						SPDLOG_DEBUG(
							"Info: {}",
							handler.get_info_log(
								compiler.get()));
						uint8_t *data;
						size_t size;
						if (handler.get_compiled_program(
							    compiler.get(),
							    &data,
							    &size) != 0) {
							throw std::runtime_error(
								"Unable to get compiled program");
						}
						std::vector<uint8_t>
							compiled_program(
								data,
								data + size);
						std::lock_guard<std::mutex>
							guard(map_lock);
						auto cached =
							this->ptx_pool
								->insert(std::make_pair(
									sha256_string,
									compiled_program))
								.first;
						compiled_ptx[name] =
							cached->second;
						SPDLOG_INFO(
							"Compile of {} done",
							name);
					}
				} catch (...) {
					std::lock_guard<std::mutex> guard(
						error_lock);
					if (!first_error)
						first_error =
							std::current_exception();
				}
			});
	}
	pool.join();
	if (first_error)
		std::rethrow_exception(first_error);
	return compiled_ptx;
}
void fatbin_record::try_loading_ptxs(class nv_attach_impl &impl)
{
	auto registration_guard = impl.lock_registration_state();
	if (ptx_loaded)
		return;
	if (impl.shared_mem_ptr == 0) {
		// CUDA fatbin/function registration can run before the agent
		// has received the shared map pointer. Leave this record
		// pending; late bootstrap will load it after the CUDA hook is
		// fully attached.
		SPDLOG_DEBUG(
			"Deferring PTX load until shared_mem_ptr is initialized");
		return;
	}
	SPDLOG_INFO("Loading & patching current fatbin..");

	auto patched = impl.hack_fatbin(original_ptx);
	if (!patched)
		throw std::runtime_error("Unable to patch PTX");
	auto patched_ptx = std::move(*patched);

	auto compiled_ptx = compile_ptxs(impl, patched_ptx);

	for (const auto &[name, ptx_and_trampoline_flag] : patched_ptx) {
		const auto &ptx = std::get<0>(ptx_and_trampoline_flag);
		bool added_trampoline = std::get<1>(ptx_and_trampoline_flag);
		const auto &compiled_elf = compiled_ptx.at(name);
		auto sha256_string =
			sha256(compiled_elf.data(), compiled_elf.size());
		if (auto itr = module_pool->find(sha256_string);
		    itr != module_pool->end()) {
			SPDLOG_INFO("Module {} found in cache", name);
			ptxs.push_back(itr->second);
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
			auto ptr = std::make_shared<ptx_in_module>(module);
			module_pool->insert(std::make_pair(sha256_string, ptr));
			ptxs.push_back(ptr);
			SPDLOG_INFO("Loaded module: {}", name);
		}
	}
	ptx_loaded = true;
}

} // namespace bpftime::attach
