#ifndef _BPFTIME_NV_ATTACH_IMPL_HPP
#define _BPFTIME_NV_ATTACH_IMPL_HPP
#include <base_attach_impl.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include <cuda.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <fstream>
#include "pos/include/workspace.h"
#include "pos/cuda_impl/remoting/workspace.h"
namespace bpftime
{
namespace attach
{

// You would replace this with your own memory reading utility.
namespace memory_utils
{
template <typename T>
bool read_memory(pid_t pid, const void *remote_addr, T *out_value)
{
	// Dummy / placeholder read. Real implementation would use
	// process_vm_readv or ptrace. Return false to indicate we haven't
	// implemented the real method here.
	return false;
}
} // namespace memory_utils

// ----------------------------------------------------------------------------
// A simple wrapper class to handle attaching to a CUDA context in another
// process. In a real scenario, you might separate this into its own .hpp/.cpp
// files.
// ----------------------------------------------------------------------------
class CUDAInjector {
    private:
	pid_t target_pid;
	CUcontext cuda_ctx{ nullptr };

	// Storing a backup of code, for illustration.
	// You can remove or adapt this if you don’t actually need code
	// injection.
	struct CodeBackup {
		CUdeviceptr addr;
		std::vector<char> original_code;
	};
	std::vector<CodeBackup> backups;

    public:
	explicit CUDAInjector(pid_t pid) : target_pid(pid)
	{
		spdlog::debug("CUDAInjector: constructor for PID {}",
			      target_pid);

		// Initialize the CUDA Driver API
		CUresult res = cuInit(0);
		if (res != CUDA_SUCCESS) {
			spdlog::error("cuInit(0) failed with code {}",
				      static_cast<int>(res));
		}
		auto ws = pos_create_workspace_cuda();

	}

	bool attach()
	{
		spdlog::info("Attaching via PTRACE to PID {}", target_pid);
		if (ptrace(PTRACE_ATTACH, target_pid, nullptr, nullptr) == -1) {
			spdlog::error("PTRACE_ATTACH failed: {}",
				      strerror(errno));
			return false;
		}
		// Wait for the process to stop
		if (waitpid(target_pid, nullptr, 0) == -1) {
			spdlog::error("waitpid failed: {}", strerror(errno));
			return false;
		}

		// Attempt to locate and set the CUDA context in the target
		// process
		if (!get_cuda_context()) {
			spdlog::error(
				"Failed to get CUDA context from process {}",
				target_pid);
			return false;
		}

		spdlog::info("Attach to PID {} successful", target_pid);
		return true;
	}

	bool detach()
	{
		spdlog::info("Detaching via PTRACE from PID {}", target_pid);
		if (ptrace(PTRACE_DETACH, target_pid, nullptr, nullptr) == -1) {
			spdlog::error("PTRACE_DETACH failed: {}",
				      strerror(errno));
			return false;
		}
		return true;
	}

    private:
	// ------------------------------------------------------------------------
	// Below is minimal logic to demonstrate how you MIGHT find a CUDA
	// context. In reality, hooking into a remote process’s memory for CUDA
	// contexts is significantly more complex (symbol lookup, driver calls,
	// etc.).
	// ------------------------------------------------------------------------
	bool get_cuda_context()
	{
		// Open the proc maps of the target
		std::ifstream mapsFile("/proc/" + std::to_string(target_pid) +
				       "/maps");
		if (!mapsFile.is_open()) {
			spdlog::error("Failed to open /proc/{}/maps",
				      target_pid);
			return false;
		}

		std::string line;
		while (std::getline(mapsFile, line)) {
			// For demo, we just check if the line references
			// 'libcuda'.
			if (line.find("libcuda.so") != std::string::npos) {
				uintptr_t start, end;
				if (sscanf(line.c_str(), "%lx-%lx", &start,
					   &end) == 2) {
					// Try reading pointers in [start, end)
					// to see if any might be a CUcontext.
					for (uintptr_t addr = start; addr < end;
					     addr += sizeof(void *)) {
						CUcontext possible_ctx;
						if (memory_utils::read_memory(
							    target_pid,
							    (void *)addr,
							    &possible_ctx)) {
							if (validate_cuda_context(
								    possible_ctx)) {
								spdlog::info(
									"Found valid CUDA context at remote address 0x{:x}",
									addr);
								cuda_ctx =
									possible_ctx;
								return true;
							}
						}
					}
				}
			}
		}
		return false;
	}

	bool validate_cuda_context(CUcontext ctx)
	{
		// Attempt to set the context in our own process. In a real
		// scenario, this doesn’t necessarily work straightforwardly
		// across processes! But for demonstration:
		CUresult res = cuCtxSetCurrent(ctx);
		if (res != CUDA_SUCCESS) {
			return false;
		}

		// If we can get a device from this context, we consider it
		// valid.
		CUdevice dev;
		res = cuCtxGetDevice(&dev);
		if (res != CUDA_SUCCESS) {
			return false;
		}
		return true;
	}
public:
	// Demonstrates how you might inject PTX or backup/restore code on the
	// fly in a remote context. This is a stub for illustration.
	bool inject_ptx(const char *ptx_code, CUdeviceptr target_addr,
			size_t code_size)
	{
		// 1. Load the PTX into a module
		CUmodule module;
		CUresult result = cuModuleLoadData(&module, ptx_code);
		if (result != CUDA_SUCCESS) {
			spdlog::error("cuModuleLoadData() failed: {}",
				      (int)result);
			return false;
		}

		// 2. Retrieve the function named "injected_kernel"
		CUfunction kernel;
		result =
			cuModuleGetFunction(&kernel, module, "injected_kernel");
		if (result != CUDA_SUCCESS) {
			spdlog::error("cuModuleGetFunction() failed: {}",
				      (int)result);
			cuModuleUnload(module);
			return false;
		}

		// 3. Backup the original code
		CodeBackup backup;
		backup.addr = target_addr;
		backup.original_code.resize(code_size);
		result = cuMemcpyDtoH(backup.original_code.data(), target_addr,
				      code_size);
		if (result != CUDA_SUCCESS) {
			spdlog::error("cuMemcpyDtoH() failed: {}", (int)result);
			cuModuleUnload(module);
			return false;
		}
		backups.push_back(backup);

		// 4. Retrieve the actual kernel code from the module’s global
		// space
		CUdeviceptr func_addr;
		size_t func_size;
		result = cuModuleGetGlobal(&func_addr, &func_size, module,
					   "injected_kernel");
		if (result != CUDA_SUCCESS) {
			spdlog::error("cuModuleGetGlobal() failed: {}",
				      (int)result);
			cuModuleUnload(module);
			return false;
		}

		// 5. Write the new code into the target location
		result = cuMemcpyDtoD(target_addr, func_addr, func_size);
		if (result != CUDA_SUCCESS) {
			spdlog::error("cuMemcpyDtoD() failed: {}", (int)result);
			cuModuleUnload(module);
			return false;
		}

		// Clean up
		cuModuleUnload(module);
		return true;
	}

	bool restore_code(CUdeviceptr addr)
	{
		for (auto &b : backups) {
			if (b.addr == addr) {
				CUresult result =
					cuMemcpyHtoD(b.addr,
						     b.original_code.data(),
						     b.original_code.size());
				return (result == CUDA_SUCCESS);
			}
		}
		return false;
	}

	bool restore_all()
	{
		bool success = true;
		for (auto &b : backups) {
			CUresult result =
				cuMemcpyHtoD(b.addr, b.original_code.data(),
					     b.original_code.size());
			if (result != CUDA_SUCCESS) {
				success = false;
			}
		}
		return success;
	}
};

extern std::optional<class nv_attach_impl *> global_nv_attach_impl;
struct nv_hooker_func_t {
	void *func;
};

struct nv_attach_private_data final : public attach_private_data {
	// The address to hook
	uint64_t addr;
	// Saved module name
	pid_t pid;
    // initialize_from_string
    int initialize_from_string(const std::string_view &sv) override;
};

constexpr int ATTACH_NV = 999;
// Used by text segment transformer to setup syscall callback
// Text segment transformer should provide a pointer to its syscall executor
// function. syscall trace attach impl will save the original syscall function,
// and replace it with one that was handled by syscall trace attach impl
extern "C" void _bpftime__setup_nv_hooker_callback(nv_hooker_func_t *hooker);

// Attach implementation of syscall trace
// It provides a callback to receive original syscall calls, and dispatch the
// concrete stuff to individual callbacks
class nv_attach_impl final : public base_attach_impl {
    public:
	// Dispatch a syscall from text transformer
	int64_t dispatch_nv(int64_t arg1, int64_t arg2, int64_t arg3,
			    int64_t arg4, int64_t arg5, int64_t arg6);
	// Set the function of calling original nv
	void set_original_nv_function(nv_hooker_func_t func)
	{
		orig_nv = func;
	}
	// Set this nv trace attach impl instance to the global ones, which
	// could be accessed by text segment transformer
	void set_to_global()
	{
		global_nv_attach_impl = this;
	}
	int detach_by_id(int id);
	int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type);
	nv_attach_impl(const nv_attach_impl &) = delete;
	nv_attach_impl &operator=(const nv_attach_impl &) = delete;
	nv_attach_impl()
	{
	}
	// Forward declare the nested Impl struct
	struct Impl;

    private:
	// The original syscall function
	nv_hooker_func_t orig_nv = { nullptr };
};

} // namespace attach
} // namespace bpftime
#endif /* _BPFTIME_NV_ATTACH_IMPL_HPP */
