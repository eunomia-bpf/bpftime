#ifndef _NV_GPU_DEVICE_MANAGER_HPP
#define _NV_GPU_DEVICE_MANAGER_HPP

#include <cuda.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "nv_attach_fatbin_record.hpp"

namespace bpftime
{
namespace attach
{

/// Per-GPU device information for multi-GPU support
struct gpu_device_info {
	int device_ordinal; // 0, 1, 2, ...
	std::string sm_arch; // "sm_86", "sm_90", etc.
	CUdevice cu_device;
	/// Per-device module pool (sha256 of ELF -> loaded CUmodule)
	std::shared_ptr<
		std::map<std::string, std::shared_ptr<ptx_in_module>>>
		module_pool;
	/// Per-device shared mem device pointer (set by runtime when
	/// CUDAContext is created)
	uintptr_t shared_mem_device_ptr = 0;
};

/// Manages enumeration and state of all GPU devices in the system.
/// Created once by nv_attach_impl and used throughout the attach lifecycle.
class gpu_device_manager {
    public:
	/// Enumerate all CUDA devices and detect their SM architectures.
	/// If BPFTIME_SM_ARCH env var is set, overrides all devices to that
	/// arch.
	void initialize();

	/// Get info for a specific device by ordinal
	gpu_device_info &get_device(int ordinal);
	const gpu_device_info &get_device(int ordinal) const;

	/// Get info for the currently active CUDA device (via cuCtxGetDevice).
	/// Falls back to device 0 if no context is active.
	gpu_device_info &get_current_device();

	/// All unique SM architectures present across all devices
	std::set<std::string> get_unique_sm_archs() const;

	/// Number of devices
	int device_count() const;

	/// Convenience: get device 0 (backward compat)
	gpu_device_info &get_default_device();

	/// Get all devices
	std::vector<gpu_device_info> &devices();
	const std::vector<gpu_device_info> &devices() const;

    private:
	std::vector<gpu_device_info> devices_;
	int count_ = 0;
};

} // namespace attach
} // namespace bpftime

#endif /* _NV_GPU_DEVICE_MANAGER_HPP */
