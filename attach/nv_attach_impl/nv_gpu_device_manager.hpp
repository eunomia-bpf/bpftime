#ifndef _NV_GPU_DEVICE_MANAGER_HPP
#define _NV_GPU_DEVICE_MANAGER_HPP

#include <cuda.h>
#include <map>
#include <memory>
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
	std::shared_ptr<std::map<std::string, std::shared_ptr<ptx_in_module>>>
		module_pool;
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

	/// Number of devices
	int device_count() const;

	/// Get all devices.
	const std::vector<gpu_device_info> &devices() const;

    private:
	std::vector<gpu_device_info> devices_;
};

} // namespace attach
} // namespace bpftime

#endif /* _NV_GPU_DEVICE_MANAGER_HPP */
