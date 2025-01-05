#include "nv_attach_impl.hpp"
#include "spdlog/spdlog.h"
#include <cerrno>
#include <optional>

#ifdef __linux__
#include <asm/unistd.h> // For architecture-specific syscall numbers
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <sys/uio.h>
#include <link.h>
#include <vector>
#include <string>
#include <fstream>
#include <cstdio>
#include <cstring>

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

// ------------------------------------------------------------------
// Implementation of the nv_attach_impl interface
// ------------------------------------------------------------------
namespace bpftime
{
namespace attach
{

// Optionally keep a global pointer if needed:
std::optional<nv_attach_impl *> global_nv_attach_impl;

// We’ll hold a unique_ptr to our CUDAInjector in nv_attach_impl
// so we can attach/detach by ID (PID) and store state.
struct nv_attach_impl::Impl {
	std::unique_ptr<CUDAInjector> injector;
};

//
// Give nv_attach_impl a private pImpl pointer, so we only
// expose the class interface in the header.
//
static inline std::unique_ptr<nv_attach_impl::Impl> &
get_impl(nv_attach_impl &obj)
{
	static std::unique_ptr<nv_attach_impl::Impl> s_impl;
	if (!s_impl) {
		s_impl = std::make_unique<nv_attach_impl::Impl>();
	}
	return s_impl;
}

int64_t nv_attach_impl::dispatch_nv(int64_t arg1, int64_t arg2, int64_t arg3,
				    int64_t arg4, int64_t arg5, int64_t arg6)
{
	spdlog::debug(
		"nv_attach_impl::dispatch_syscall({}, {}, {}, {}, {}, {})",
		arg1, arg2, arg3, arg4, arg5, arg6);

	// For now, just returning 0 or an error code as a placeholder.
	// Real logic might do PTRACE_SYSCALL on the attached process, etc.
	return 0;
}

int nv_attach_impl::detach_by_id(int id)
{
	// Suppose `id` is actually the PID we attached. We can reuse the same
	// pImpl.
	auto &implPtr = get_impl(*this);
	if (!implPtr->injector) {
		spdlog::warn("detach_by_id({}): no active injector!", id);
		return -1;
	}

	spdlog::info("nv_attach_impl::detach_by_id({})", id);
	if (!implPtr->injector->detach()) {
		spdlog::error("Failed to detach from PID {}", id);
		return -1;
	}

	spdlog::info("Detached from PID {}", id);
	implPtr->injector.reset();
	return 0;
}

int nv_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&cb, const attach_private_data &private_data,
	int attach_type)
{
	spdlog::info(
		"nv_attach_impl::create_attach_with_ebpf_callback(...) attach_type={}",
		attach_type);

	// For example, we interpret private_data.pid as the target PID:
	pid_t target_pid =
		static_cast<const nv_attach_private_data &>(private_data).pid;

	// Create our injector if not yet created
	auto &implPtr = get_impl(*this);
	implPtr->injector = std::make_unique<CUDAInjector>(target_pid);

	// Attempt to attach
	if (!implPtr->injector->attach()) {
		spdlog::error("Failed to attach to PID {}", target_pid);
		return -1;
	}

	// If we have an eBPF callback, call it here
	if (cb) {
		// For example:
		void *arg1 = nullptr;
		unsigned long arg2 = 0;
		unsigned long *arg3 = nullptr;

		int rc = cb(arg1, arg2, arg3);
		spdlog::info("ebpf_run_callback returned {}", rc);
	}

	return 0;
}

// A C interface for hooking up your own system-call hooking logic
extern "C" void _bpftime__setup_nv_hooker_callback(nv_hooker_func_t *hooker)
{
	spdlog::info("_bpftime__setup_nv_hooker_callback(...) called");
	// Example: store the callback pointer if you want
	// In real usage, you might store it globally or pass it somewhere else.
}

} // namespace attach
} // namespace bpftime
